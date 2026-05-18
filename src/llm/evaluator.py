"""モデル別の評価リクエストを管理する Evaluator."""

import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel

from src.evaluation.models import EvaluationCriteria
from src.llm.client import CacheHandle, LLMClient, ModelConfig, PromptTemplates
from src.llm.factory import build_client
from src.llm.prompt_loader import load_prompt_templates


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
        templates: PromptTemplates | None = None,
    ) -> None:
        """初期化.

        Args:
            config: アプリケーション設定辞書（path.env, llm.prompt_yml を読む）
            model_config: 使用するモデルの設定
            client: 既に構築済みの LLMClient（テスト用）。
                    None の場合は factory で生成する。
            templates: 既にロード済みの PromptTemplates。
                    None の場合は config から読み込む。
        """
        env_path_str = config.get("path", {}).get("env")
        if env_path_str:
            env_path = Path(env_path_str)
            if env_path.is_file():
                load_dotenv(env_path)

        self._templates = templates if templates is not None else load_prompt_templates(
            config
        )
        self.model_config = model_config
        self._client: LLMClient = (
            client if client is not None else build_client(model_config)
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
            criterion_name=criteria.name,
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
