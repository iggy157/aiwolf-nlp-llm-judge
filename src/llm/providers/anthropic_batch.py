"""Anthropic Message Batches API クライアント.

フロー:
    1. リクエスト群を MessageBatch.create の requests に変換（cache_control 付き）
    2. ポーリング（messages.batches.retrieve）
    3. ended になったら messages.batches.results() でストリーミング取得
    4. tool_use ブロックを抽出して Pydantic 化

24h SLA, 50% 割引。cache_control は通常リクエストと同じ書式で OK。
"""

import logging
import os
import time

from anthropic import Anthropic
from pydantic import BaseModel

from src.llm.batch import BatchClient, BatchRequest, BatchResult
from src.llm.client import ModelConfig, PromptTemplates
from src.llm.prompt_renderer import render_system, split_user_prompt
from src.llm.providers.anthropic_client import (
    DEFAULT_MAX_TOKENS,
    EVALUATION_TOOL_NAME,
)

logger = logging.getLogger(__name__)


class AnthropicBatchClient(BatchClient):
    """Anthropic Message Batches API クライアント."""

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

    def supports_batch(self) -> bool:
        return True

    def submit_and_wait(
        self,
        requests: list[BatchRequest],
        templates: PromptTemplates,
        output_structure: type[BaseModel],
        poll_interval_seconds: float = 60.0,
        max_wait_seconds: float = 86400.0,
    ) -> list[BatchResult]:
        if not requests:
            return []

        params_list = self._build_requests(requests, templates, output_structure)
        logger.info(
            f"[{self.model_config.id}] submitting {len(params_list)} requests to "
            "Anthropic batch API"
        )

        batch = self._client.messages.batches.create(requests=params_list)
        logger.info(
            f"[{self.model_config.id}] batch created: id={batch.id}, "
            f"processing_status={batch.processing_status}"
        )

        self._poll_until_ended(batch.id, poll_interval_seconds, max_wait_seconds)

        return self._collect_results(batch.id, output_structure)

    def _build_requests(
        self,
        requests: list[BatchRequest],
        templates: PromptTemplates,
        output_structure: type[BaseModel],
    ) -> list[dict]:
        system_text = render_system(templates)
        schema = output_structure.model_json_schema()
        tool = {
            "name": EVALUATION_TOOL_NAME,
            "description": "Submit the evaluation result for the given criteria.",
            "input_schema": schema,
        }
        max_tokens = int(self.model_config.extra.get("max_tokens", DEFAULT_MAX_TOKENS))

        params_list: list[dict] = []
        for req in requests:
            prefix_text, varying_text = split_user_prompt(
                templates,
                req.character_info,
                req.criteria_description,
                req.log_json,
            )
            params_list.append(
                {
                    "custom_id": req.custom_id,
                    "params": {
                        "model": self.model_config.model,
                        "max_tokens": max_tokens,
                        "system": [
                            {
                                "type": "text",
                                "text": system_text,
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": prefix_text,
                                        "cache_control": {"type": "ephemeral"},
                                    },
                                    {"type": "text", "text": varying_text},
                                ],
                            }
                        ],
                        "tools": [tool],
                        "tool_choice": {"type": "tool", "name": EVALUATION_TOOL_NAME},
                    },
                }
            )
        return params_list

    def _poll_until_ended(self, batch_id: str, poll_interval: float, max_wait: float):
        deadline = time.time() + max_wait
        last_status = None
        while True:
            batch = self._client.messages.batches.retrieve(batch_id)
            if batch.processing_status != last_status:
                logger.info(
                    f"[{self.model_config.id}] batch {batch_id} status: "
                    f"{batch.processing_status}"
                )
                last_status = batch.processing_status
            if batch.processing_status == "ended":
                return batch
            if time.time() > deadline:
                raise TimeoutError(
                    f"[{self.model_config.id}] batch {batch_id} did not end within "
                    f"{max_wait}s (status={batch.processing_status})"
                )
            time.sleep(poll_interval)

    def _collect_results(
        self, batch_id: str, output_structure: type[BaseModel]
    ) -> list[BatchResult]:
        results: list[BatchResult] = []
        for item in self._client.messages.batches.results(batch_id):
            custom_id = getattr(item, "custom_id", "?")
            result = getattr(item, "result", None)
            if result is None:
                results.append(
                    BatchResult(custom_id=custom_id, success=False, error="no result")
                )
                continue

            result_type = getattr(result, "type", None)
            if result_type == "succeeded":
                results.append(
                    self._parse_succeeded(custom_id, result, output_structure)
                )
            elif result_type == "errored":
                error_obj = getattr(result, "error", None)
                error_type = (
                    getattr(error_obj, "type", "unknown")
                    if error_obj is not None
                    else "unknown"
                )
                error_msg = (
                    getattr(error_obj, "message", str(error_obj))
                    if error_obj is not None
                    else "unknown error"
                )
                results.append(
                    BatchResult(
                        custom_id=custom_id,
                        success=False,
                        error=f"API error [{error_type}]: {error_msg}",
                    )
                )
            elif result_type in ("expired", "canceled"):
                results.append(
                    BatchResult(
                        custom_id=custom_id,
                        success=False,
                        error=f"request {result_type}",
                    )
                )
            else:
                results.append(
                    BatchResult(
                        custom_id=custom_id,
                        success=False,
                        error=f"unknown result type: {result_type!r}",
                    )
                )
        return results

    def _parse_succeeded(
        self, custom_id: str, result, output_structure: type[BaseModel]
    ) -> BatchResult:
        message = getattr(result, "message", None)
        if message is None:
            return BatchResult(
                custom_id=custom_id,
                success=False,
                error="succeeded but no message",
            )
        try:
            for block in message.content:
                if (
                    getattr(block, "type", None) == "tool_use"
                    and block.name == EVALUATION_TOOL_NAME
                ):
                    parsed = output_structure.model_validate(block.input)
                    return BatchResult(
                        custom_id=custom_id, success=True, response=parsed
                    )
            raise ValueError(f"no tool_use block named '{EVALUATION_TOOL_NAME}'")
        except Exception as e:
            return BatchResult(custom_id=custom_id, success=False, error=str(e))
