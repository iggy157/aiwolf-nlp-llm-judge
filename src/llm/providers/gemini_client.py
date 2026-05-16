"""Google Gemini 用クライアント.

provider: gemini    -> Google AI Studio API（api_key で認証）
provider: vertex_ai -> Vertex AI（project_id + location、GOOGLE_APPLICATION_CREDENTIALS でADC認証）

Prompt Caching:
    1ゲーム単位で `client.caches.create()` を呼んで system + character_info を
    CachedContent リソースとして保存し、各評価基準呼び出しでは
    GenerateContentConfig.cached_content にリソース名を渡す。
    ゲーム終了時に `client.caches.delete()` で解放する。
    TTL は ModelConfig.cache_ttl_seconds（デフォルト3600秒）。
"""

import logging
import os

from google import genai
from google.genai import types
from pydantic import BaseModel

from src.llm.client import CacheHandle, ModelConfig, PromptTemplates
from src.llm.prompt_renderer import (
    EVALUATION_BLOCK_HEADER,
    LOG_BLOCK_HEADER,
    render_system,
    render_user,
    split_user_prompt,
)

logger = logging.getLogger(__name__)


class GeminiClient:
    """Gemini 用クライアント（AI Studio / Vertex AI 共通）."""

    AI_STUDIO_PROVIDER = "gemini"
    VERTEX_PROVIDER = "vertex_ai"

    def __init__(self, model_config: ModelConfig) -> None:
        self.model_config = model_config
        self._client = self._build_client(model_config)

    @staticmethod
    def _build_client(model_config: ModelConfig) -> genai.Client:
        if model_config.provider == GeminiClient.VERTEX_PROVIDER:
            if not model_config.project_id or not model_config.location:
                raise ValueError(
                    f"{model_config.id}: project_id and location are required "
                    "for vertex_ai provider"
                )
            return genai.Client(
                vertexai=True,
                project=model_config.project_id,
                location=model_config.location,
            )
        if not model_config.api_key_env:
            raise ValueError(
                f"{model_config.id}: api_key_env is required for gemini provider"
            )
        api_key = os.environ.get(model_config.api_key_env)
        if not api_key:
            raise ValueError(
                f"{model_config.id}: env var {model_config.api_key_env} is not set"
            )
        return genai.Client(api_key=api_key)

    def open_cache(
        self, character_info: str, templates: PromptTemplates
    ) -> CacheHandle | None:
        """system + character_info を CachedContent リソースとして保存."""
        system_text = render_system(templates)
        # 評価基準とログを抜いた prefix の content を組む（split_user_prompt の prefix 側）
        prefix_text, _ = split_user_prompt(
            templates, character_info, "", ""
        )

        try:
            cache = self._client.caches.create(
                model=self.model_config.model,
                config=types.CreateCachedContentConfig(
                    system_instruction=system_text,
                    contents=[
                        types.Content(
                            role="user",
                            parts=[types.Part.from_text(text=prefix_text)],
                        )
                    ],
                    ttl=f"{int(self.model_config.cache_ttl_seconds)}s",
                ),
            )
        except Exception as e:
            # キャッシュ閾値（モデルにより 1024〜4096 tokens）未満などで失敗することがある。
            # その場合はキャッシュなしで動くようにフォールバック。
            logger.warning(
                f"[{self.model_config.id}] Failed to create cached content "
                f"(falling back to no-cache mode): {e}"
            )
            return None

        return CacheHandle(provider=self.model_config.provider, resource_name=cache.name)

    def close_cache(self, handle: CacheHandle) -> None:
        if not handle.resource_name:
            return
        try:
            self._client.caches.delete(name=handle.resource_name)
        except Exception as e:
            # PERMISSION_DENIED / QUOTA_EXCEEDED / 認証エラーは「気付くべき」事案。
            # transient なエラー（NOT_FOUND: TTLで既に消えた等）は warning にとどめる。
            msg = str(e)
            severe = any(
                token in msg
                for token in (
                    "PERMISSION_DENIED",
                    "UNAUTHENTICATED",
                    "QUOTA",
                    "RESOURCE_EXHAUSTED",
                )
            )
            if severe:
                logger.error(
                    f"[{self.model_config.id}] Failed to delete cached content "
                    f"{handle.resource_name} (severe): {e}"
                )
            else:
                logger.warning(
                    f"[{self.model_config.id}] Failed to delete cached content "
                    f"{handle.resource_name}: {e}"
                )

    def evaluate(
        self,
        criteria_description: str,
        character_info: str,
        log_json: str,
        templates: PromptTemplates,
        output_structure: type[BaseModel],
        cache_handle: CacheHandle | None = None,
    ) -> BaseModel:
        """Gemini を response_schema 強制で呼び出して構造化レスポンスを返す."""
        if cache_handle is not None and cache_handle.resource_name:
            # キャッシュ使用時は prefix を送らず criteria+log のみ送る
            user_text = (
                f"{EVALUATION_BLOCK_HEADER}\n\n{criteria_description}\n\n"
                f"{LOG_BLOCK_HEADER}\n\n{log_json}"
            )
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=output_structure,
                cached_content=cache_handle.resource_name,
            )
        else:
            system_text = render_system(templates)
            user_text = render_user(
                templates, character_info, criteria_description, log_json
            )
            config = types.GenerateContentConfig(
                system_instruction=system_text,
                response_mime_type="application/json",
                response_schema=output_structure,
            )

        response = self._client.models.generate_content(
            model=self.model_config.model,
            contents=user_text,
            config=config,
        )

        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, output_structure):
            return parsed

        text = getattr(response, "text", None)
        if not text:
            raise ValueError(
                f"{self.model_config.id}: Gemini returned no parsed object and no text"
            )
        return output_structure.model_validate_json(text)
