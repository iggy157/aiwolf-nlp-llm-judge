"""モデル別の評価リクエストを管理する Evaluator."""

import json
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

from src.evaluation.models import EvaluationCriteria
from src.llm.client import CacheHandle, LLMClient, ModelConfig, PromptTemplates
from src.llm.factory import build_client


class Evaluator:
    """LLMClient に評価リクエストを委譲する薄いラッパ.

    プロンプトテンプレートの読み込みと、評価基準/ログを文字列に整形して
    LLMClient.evaluate に渡す責務だけを持つ。
    """

    def __init__(
        self,
        config: dict[str, Any],
        model_config: ModelConfig,
        client: LLMClient | None = None,
    ) -> None:
        """初期化.

        Args:
            config: アプリケーション設定辞書（path.env, llm.prompt_yml を読む）
            model_config: 使用するモデルの設定
            client: 既に構築済みの LLMClient（テスト用）。
                    None の場合は factory で生成する。
        """
        try:
            env_path = Path(config["path"]["env"])
            prompt_yml_path = Path(config["llm"]["prompt_yml"])
        except KeyError as e:
            raise KeyError(f"必要な設定キーが見つかりません: {e}")

        if env_path.is_file():
            load_dotenv(env_path)

        if not prompt_yml_path.is_file():
            raise FileNotFoundError(
                f"プロンプトYAMLファイルが見つかりません: {prompt_yml_path}"
            )

        try:
            with open(prompt_yml_path, "r", encoding="utf-8") as f:
                prompt_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"プロンプトYAMLファイルの解析に失敗しました: {e}")

        # prompts.yaml では旧来 "developer" キーをシステムプロンプトに使っているため、
        # それを system テンプレートとして扱う。
        system_template = prompt_data.get("developer") or prompt_data.get("system")
        user_template = prompt_data.get("user")
        if not system_template or not user_template:
            raise ValueError(
                "prompts.yaml には 'developer'（または 'system'）と 'user' の両方が必要です"
            )

        self._templates = PromptTemplates(system=system_template, user=user_template)
        self.model_config = model_config
        self._client: LLMClient = client if client is not None else build_client(
            model_config
        )

    def evaluation(
        self,
        criteria: EvaluationCriteria,
        log: list[dict[str, Any]],
        output_structure: type[BaseModel],
        character_info: str = "",
        cache_handle: CacheHandle | None = None,
    ) -> BaseModel:
        """1評価基準に対するLLM呼び出し."""
        return self._client.evaluate(
            criteria_description=criteria.description,
            character_info=character_info,
            log_json=json.dumps(log, ensure_ascii=False),
            templates=self._templates,
            output_structure=output_structure,
            cache_handle=cache_handle,
        )

    @property
    def templates(self) -> PromptTemplates:
        return self._templates

    def open_cache(self, character_info: str) -> CacheHandle | None:
        """ゲーム単位で prefix キャッシュを準備."""
        return self._client.open_cache(character_info, self._templates)

    def close_cache(self, handle: CacheHandle | None) -> None:
        if handle is None:
            return
        self._client.close_cache(handle)
