"""バッチAPI 用の共通インターフェースとデータ型.

各クラウドプロバイダ（OpenAI / Anthropic / Gemini / Vertex）は
バッチ APIs を持つが、それぞれ投入形式・ライフサイクル・結果取得方法が異なる。
本モジュールでは「リクエスト群を投入 → ポーリング → 結果取得」までを
プロバイダ非依存に扱えるよう抽象化する。

ローカルプロバイダ（openai_compatible）はバッチAPI非対応のため、
BatchOrchestrator 側で同期パスにフォールバックする。
"""

from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel

from src.llm.client import ModelConfig, PromptTemplates


@dataclass(frozen=True)
class BatchRequest:
    """バッチ投入する1リクエスト."""

    custom_id: str  # 結果照合用のユニークID（例: "<game_id>:<criteria_name>"）
    criteria_description: str
    character_info: str
    log_json: str


@dataclass
class BatchResult:
    """バッチ結果の1件."""

    custom_id: str
    success: bool
    response: BaseModel | None = None  # 成功時の Pydantic レスポンス
    error: str | None = None  # 失敗時のメッセージ


class BatchClient(Protocol):
    """プロバイダ別バッチAPIクライアントのインターフェース."""

    model_config: ModelConfig

    def supports_batch(self) -> bool:
        """このクライアントがバッチAPIをサポートするか.

        ローカル推論サーバ（openai_compatible）は False を返し、
        オーケストレータ側で同期パスにフォールバックされる。
        """
        ...

    def submit_and_wait(
        self,
        requests: list[BatchRequest],
        templates: PromptTemplates,
        output_structure: type[BaseModel],
        poll_interval_seconds: float = 60.0,
        max_wait_seconds: float = 86400.0,
    ) -> list[BatchResult]:
        """全リクエストを投入し、完了まで待ってから結果を返す.

        Args:
            requests: バッチ投入するリクエスト
            templates: プロンプトテンプレート
            output_structure: 期待する出力Pydanticモデルクラス
            poll_interval_seconds: ステータスチェック間隔
            max_wait_seconds: タイムアウト

        Returns:
            BatchResult のリスト（順序はリクエスト順とは限らない、custom_id で照合する）
        """
        ...


@dataclass
class BatchProgress:
    """進捗表示用."""

    model_id: str
    total: int
    completed: int = 0
    failed: int = 0

    @property
    def remaining(self) -> int:
        return self.total - self.completed - self.failed
