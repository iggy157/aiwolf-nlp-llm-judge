"""Anthropic Claude 用クライアント.

構造化出力は tool_use（強制ツール呼び出し）を利用する。
レスポンスから tool_use ブロックを取り出して Pydantic モデルに復元。

Prompt Caching:
    全評価基準で共通の prefix（system プロンプト + character_info）に
    `cache_control: {type: ephemeral}` マーカーを付与する。
    Anthropic 側で 5 分 TTL のキャッシュとして自動的に再利用される。
    キャッシュサイズの閾値（1024 tokens for Sonnet 等）を下回る場合は
    マーカーが付いていても課金/動作に影響しない。
"""

import os

from anthropic import Anthropic
from pydantic import BaseModel

from src.llm.client import CacheHandle, ModelConfig, PromptTemplates


EVALUATION_TOOL_NAME = "submit_evaluation"
DEFAULT_MAX_TOKENS = 8192


class AnthropicClient:
    """Anthropic Claude 用クライアント."""

    def __init__(self, model_config: ModelConfig) -> None:
        self.model_config = model_config
        if not model_config.api_key_env:
            raise ValueError(
                f"{model_config.id}: api_key_env is required for Anthropic provider"
            )
        api_key = os.environ.get(model_config.api_key_env)
        if not api_key:
            raise ValueError(
                f"{model_config.id}: env var {model_config.api_key_env} is not set"
            )
        self._client = Anthropic(
            api_key=api_key,
            timeout=model_config.request_timeout,
        )

    def open_cache(
        self, character_info: str, templates: PromptTemplates
    ) -> CacheHandle | None:
        """Anthropic は cache_control を毎回付与するだけなので明示ハンドルは不要."""
        return None

    def close_cache(self, handle: CacheHandle) -> None:
        return None

    def evaluate(
        self,
        criteria_description: str,
        character_info: str,
        log_json: str,
        templates: PromptTemplates,
        output_structure: type[BaseModel],
        cache_handle: CacheHandle | None = None,
    ) -> BaseModel:
        """Claude を tool_use で呼び出して構造化レスポンスを返す."""
        from jinja2 import Template

        system_text = Template(templates.system).render().strip()

        # user 側プロンプトを「キャッシュ対象（character_info まで）」と
        # 「キャッシュ対象外（criteria + log）」に分割。
        prefix_text, varying_text = self._split_user_prompt(
            templates.user, character_info, criteria_description, log_json
        )

        # system はリスト形式で渡し、最後のブロックに cache_control を付ける。
        system_blocks = [
            {
                "type": "text",
                "text": system_text,
                "cache_control": {"type": "ephemeral"},
            }
        ]

        user_content = [
            {
                "type": "text",
                "text": prefix_text,
                "cache_control": {"type": "ephemeral"},
            },
            {"type": "text", "text": varying_text},
        ]

        schema = output_structure.model_json_schema()
        tool = {
            "name": EVALUATION_TOOL_NAME,
            "description": "Submit the evaluation result for the given criteria.",
            "input_schema": schema,
        }

        max_tokens = int(self.model_config.extra.get("max_tokens", DEFAULT_MAX_TOKENS))

        response = self._client.messages.create(
            model=self.model_config.model,
            max_tokens=max_tokens,
            system=system_blocks,
            messages=[{"role": "user", "content": user_content}],
            tools=[tool],
            tool_choice={"type": "tool", "name": EVALUATION_TOOL_NAME},
        )

        for block in response.content:
            if (
                getattr(block, "type", None) == "tool_use"
                and block.name == EVALUATION_TOOL_NAME
            ):
                return output_structure.model_validate(block.input)

        raise ValueError(
            f"{self.model_config.id}: no tool_use block named "
            f"'{EVALUATION_TOOL_NAME}' in Claude response"
        )

    @staticmethod
    def _split_user_prompt(
        user_template: str,
        character_info: str,
        criteria_description: str,
        log_json: str,
    ) -> tuple[str, str]:
        """ユーザープロンプトをキャッシュ可能 prefix と可変部に分割.

        prompts.yaml の user テンプレートは
        「説明文 + character_info + criteria_description + log」の順で並んでいる前提。
        criteria_description より前を prefix とする。
        """
        from jinja2 import Template

        # 1) 可変部だけ空にして prefix を取り出す
        prefix_template = Template(user_template)
        prefix_text = prefix_template.render(
            character_info=character_info,
            criteria_description="",
            log="",
        ).rstrip()

        # 2) 可変部
        varying_text = (
            f"## 評価基準\n\n{criteria_description}\n\n"
            f"## 評価対象のログ\n\n{log_json}"
        )
        return prefix_text, varying_text
