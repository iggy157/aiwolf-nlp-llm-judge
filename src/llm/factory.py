"""LLMClient ファクトリ.

設定（ModelConfig）から適切なプロバイダ実装をインスタンス化する。
"""

from typing import Any

from src.llm.batch import BatchClient
from src.llm.client import LLMClient, ModelConfig
from src.llm.providers import (
    AnthropicBatchClient,
    AnthropicClient,
    GeminiBatchClient,
    GeminiClient,
    OpenAIBatchClient,
    OpenAICompatibleClient,
)


# provider 名 -> 同期クライアントクラス
_PROVIDER_REGISTRY: dict[str, type] = {
    "openai": OpenAICompatibleClient,
    "openai_compatible": OpenAICompatibleClient,
    "anthropic": AnthropicClient,
    "gemini": GeminiClient,
    "vertex_ai": GeminiClient,
}

# provider 名 -> バッチクライアントクラス（バッチAPI非対応の場合は None）
_BATCH_REGISTRY: dict[str, type | None] = {
    "openai": OpenAIBatchClient,
    "openai_compatible": OpenAIBatchClient,  # supports_batch()=False で同期にフォールバック
    "anthropic": AnthropicBatchClient,
    "gemini": GeminiBatchClient,
    "vertex_ai": GeminiBatchClient,
}


def supported_providers() -> list[str]:
    """サポート対象のプロバイダ名一覧."""
    return sorted(_PROVIDER_REGISTRY.keys())


def build_client(model_config: ModelConfig) -> LLMClient:
    """ModelConfig から同期 LLMClient を生成."""
    cls = _PROVIDER_REGISTRY.get(model_config.provider)
    if cls is None:
        raise ValueError(
            f"Unknown LLM provider '{model_config.provider}'. "
            f"Supported: {supported_providers()}"
        )
    return cls(model_config)


def build_batch_client(model_config: ModelConfig) -> BatchClient | None:
    """ModelConfig からバッチクライアントを生成. バッチAPI非対応の場合は None."""
    cls = _BATCH_REGISTRY.get(model_config.provider)
    if cls is None:
        return None
    client = cls(model_config)
    if not client.supports_batch():
        return None
    return client


def model_config_from_dict(data: dict[str, Any]) -> ModelConfig:
    """設定辞書（settings.yaml の llm.models の1要素）から ModelConfig を作成.

    必須キー: id, provider, model
    任意キー: api_key_env, base_url, project_id, location, system_role,
              request_timeout, extra
    """
    required = ("id", "provider", "model")
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Model config missing required keys: {missing}")

    provider = data["provider"]
    if provider not in _PROVIDER_REGISTRY:
        raise ValueError(
            f"Unknown LLM provider '{provider}' in model '{data['id']}'. "
            f"Supported: {supported_providers()}"
        )

    return ModelConfig(
        id=data["id"],
        provider=provider,
        model=data["model"],
        api_key_env=data.get("api_key_env"),
        base_url=data.get("base_url"),
        project_id=data.get("project_id"),
        location=data.get("location"),
        gcs_bucket=data.get("gcs_bucket"),
        system_role=data.get("system_role", "system"),
        request_timeout=float(data.get("request_timeout", 600.0)),
        cache_ttl_seconds=int(data.get("cache_ttl_seconds", 3600)),
        extra=dict(data.get("extra", {})),
    )
