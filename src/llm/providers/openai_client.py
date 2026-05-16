"""OpenAI および OpenAI互換エンドポイント用クライアント.

provider: openai            -> OpenAI 公式（structured output は .parse() を使用）
provider: openai_compatible -> vLLM / Ollama / llama.cpp / LM Studio など
                               （response_format に json_schema を渡す）
"""

import json
import os

from openai import OpenAI
from pydantic import BaseModel

from src.llm.client import CacheHandle, ModelConfig, PromptTemplates


class OpenAICompatibleClient:
    """OpenAI公式 + OpenAI互換ローカル推論サーバを統合的に扱うクライアント."""

    OFFICIAL_PROVIDER = "openai"

    def __init__(self, model_config: ModelConfig) -> None:
        """初期化.

        Args:
            model_config: モデル設定
        """
        self.model_config = model_config
        self._client = self._build_client(model_config)
        # OpenAI 公式の場合は .parse() を使い、互換APIでは json_schema 経由で取得する
        self._use_native_parse = model_config.provider == self.OFFICIAL_PROVIDER

    @staticmethod
    def _build_client(model_config: ModelConfig) -> OpenAI:
        """設定から OpenAI クライアントを構築.

        APIキーは env から取得。base_url が指定されていればそちらに切り替える。
        """
        api_key: str | None = None
        if model_config.api_key_env:
            api_key = os.environ.get(model_config.api_key_env)
            if not api_key:
                # ローカルサーバ向けにダミーキーを許容
                api_key = "EMPTY"

        kwargs: dict = {
            "api_key": api_key or "EMPTY",
            "timeout": model_config.request_timeout,
        }
        if model_config.base_url:
            kwargs["base_url"] = model_config.base_url
        return OpenAI(**kwargs)

    def open_cache(
        self, character_info: str, templates: PromptTemplates
    ) -> CacheHandle | None:
        """OpenAI / OpenAI互換ローカルサーバは自動キャッシュ（明示制御不要）."""
        return None

    def close_cache(self, handle: CacheHandle) -> None:
        """OpenAI / OpenAI互換は何もしない."""
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
        """評価を実行して構造化レスポンスを返す."""
        from jinja2 import Template

        system_text = Template(templates.system).render().strip()
        user_text = (
            Template(templates.user)
            .render(
                character_info=character_info,
                criteria_description=criteria_description,
                log=log_json,
            )
            .strip()
        )

        messages = [
            {"role": self.model_config.system_role, "content": system_text},
            {"role": "user", "content": user_text},
        ]

        if self._use_native_parse:
            return self._parse_native(messages, output_structure)
        return self._parse_compatible(messages, output_structure)

    def _parse_native(
        self, messages: list[dict], output_structure: type[BaseModel]
    ) -> BaseModel:
        """OpenAI 公式の .parse() を使った構造化出力."""
        response = self._client.beta.chat.completions.parse(
            model=self.model_config.model,
            messages=messages,
            response_format=output_structure,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise ValueError("OpenAI returned no parsed content")
        return parsed

    def _parse_compatible(
        self, messages: list[dict], output_structure: type[BaseModel]
    ) -> BaseModel:
        """OpenAI互換エンドポイント用に json_schema を直接指定して取得."""
        schema = output_structure.model_json_schema()
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": output_structure.__name__,
                "schema": schema,
                "strict": True,
            },
        }

        kwargs: dict = {
            "model": self.model_config.model,
            "messages": messages,
            "response_format": response_format,
        }
        if self.model_config.extra:
            kwargs["extra_body"] = self.model_config.extra

        response = self._client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        if not content:
            raise ValueError(
                f"{self.model_config.id}: server returned empty response content"
            )

        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"{self.model_config.id}: response was not valid JSON: {e}"
            ) from e

        return output_structure.model_validate(data)
