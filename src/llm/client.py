"""LLMクライアントの共通インターフェース."""

from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel


@dataclass(frozen=True)
class ModelConfig:
    """単一モデルの実行設定.

    settings.yaml の llm.models 配列の1要素を表す。

    Attributes:
        id: ユーザーが識別に使う任意の文字列（出力ディレクトリ名にもなる）
        provider: openai | openai_compatible | anthropic | gemini | vertex_ai
        model: APIに渡すモデル名（例: gpt-4o, claude-sonnet-4-5, gemini-2.5-pro）
        api_key_env: APIキーを格納する環境変数名（不要な場合はNone）
        base_url: OpenAI互換エンドポイントURL（ローカル推論サーバ用）
        project_id: Vertex AI のGCPプロジェクトID
        location: Vertex AI のロケーション
        gcs_bucket: Vertex AI Batch API 用のGCSバケットURI（例: gs://my-bucket/aiwolf-judge/）
        system_role: システムメッセージのロール名（"system" or "developer"）
        request_timeout: リクエストタイムアウト（秒）
        cache_ttl_seconds: Gemini/Vertex の CachedContent TTL（秒）。デフォルト1時間
        extra: プロバイダ固有の追加パラメータ（vLLMのguided_decoding等）
    """

    id: str
    provider: str
    model: str
    api_key_env: str | None = None
    base_url: str | None = None
    project_id: str | None = None
    location: str | None = None
    gcs_bucket: str | None = None
    system_role: str = "system"
    request_timeout: float = 600.0
    cache_ttl_seconds: int = 3600
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PromptTemplates:
    """プロンプトテンプレート（developer/system + user）."""

    system: str  # config/prompts.yaml の 'developer' キーの内容
    user: str


@dataclass
class CacheHandle:
    """プロバイダ固有のキャッシュリソース参照.

    Gemini/Vertex のように明示的なキャッシュリソースを使うプロバイダは
    `resource_name` にキャッシュ名（"cachedContents/xxx" 等）を入れる。
    キャッシュを使わない/自動キャッシュのプロバイダは None ハンドルで運用。
    """

    provider: str
    resource_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class LLMClient(Protocol):
    """LLMクライアントの共通インターフェース.

    各プロバイダ実装はこのプロトコルを満たす。
    並列処理（ThreadPoolExecutor）から呼ばれるため、`evaluate` はスレッドセーフであること。
    """

    model_config: ModelConfig

    def evaluate(
        self,
        criteria_description: str,
        character_info: str,
        log_json: str,
        templates: PromptTemplates,
        output_structure: type[BaseModel],
        cache_handle: CacheHandle | None = None,
    ) -> BaseModel:
        """評価リクエストを送り、構造化された結果を返す.

        Args:
            criteria_description: 評価基準の説明文
            character_info: キャラクター設定情報
            log_json: 評価対象のログ（JSON文字列）
            templates: プロンプトテンプレート
            output_structure: 期待する出力Pydanticモデルクラス
            cache_handle: open_cache で取得したキャッシュハンドル（プロバイダによっては無視）

        Returns:
            output_structure 型のインスタンス

        Raises:
            ValueError: レスポンス解析やバリデーションに失敗した場合
        """
        ...

    def open_cache(
        self,
        character_info: str,
        templates: PromptTemplates,
    ) -> CacheHandle | None:
        """1ゲーム分の prefix（system + character_info）をキャッシュ.

        プロバイダがキャッシュを必要としない場合は None を返す。
        Anthropic は cache_control マーカーを直接 evaluate 内で適用するためここでは None。
        Gemini/Vertex は CachedContent リソースを作成する。

        Returns:
            CacheHandle またはキャッシュ未使用なら None
        """
        ...

    def close_cache(self, handle: CacheHandle) -> None:
        """open_cache で確保したリソースを解放."""
        ...
