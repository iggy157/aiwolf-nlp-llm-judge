"""OpenAI Batch API クライアント.

フロー:
    1. リクエスト群を JSONL に変換
    2. Files API にアップロード（purpose='batch'）
    3. Batches API で job 作成
    4. 完了までポーリング
    5. output_file_id から JSONL をダウンロード
    6. JSONL を BatchResult に変換

24h SLA, 50% 割引。Prompt Caching は OpenAI 側で自動。
"""

import io
import json
import logging
import os
import time

from openai import OpenAI
from pydantic import BaseModel

from src.llm.batch import BatchClient, BatchRequest, BatchResult
from src.llm.client import ModelConfig, PromptTemplates
from src.llm.prompt_renderer import render_system, render_user

logger = logging.getLogger(__name__)


class OpenAIBatchClient(BatchClient):
    """OpenAI Batch API クライアント.

    provider 文字列が 'openai' または 'openai_compatible' のとき使われるが、
    互換サーバ（vLLM等）はバッチAPI非対応のため supports_batch() で除外する。
    """

    def __init__(self, model_config: ModelConfig) -> None:
        self.model_config = model_config
        self._client = self._build_client(model_config)

    @staticmethod
    def _build_client(model_config: ModelConfig) -> OpenAI:
        api_key = None
        if model_config.api_key_env:
            api_key = os.environ.get(model_config.api_key_env)
        kwargs: dict = {
            "api_key": api_key or "EMPTY",
            "timeout": model_config.request_timeout,
        }
        if model_config.base_url:
            kwargs["base_url"] = model_config.base_url
        return OpenAI(**kwargs)

    def supports_batch(self) -> bool:
        # ローカル推論サーバ（base_url 指定）は Batch API 非対応
        return self.model_config.provider == "openai"

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

        jsonl_bytes = self._build_input_jsonl(requests, templates, output_structure)
        logger.info(
            f"[{self.model_config.id}] uploading batch input "
            f"({len(jsonl_bytes)} bytes, {len(requests)} requests)"
        )

        file_obj = self._client.files.create(
            file=("batch_input.jsonl", io.BytesIO(jsonl_bytes)),
            purpose="batch",
        )
        batch = self._client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"source": "aiwolf-nlp-llm-judge", "model_id": self.model_config.id},
        )
        logger.info(
            f"[{self.model_config.id}] batch created: id={batch.id}, "
            f"status={batch.status}"
        )

        completed = self._poll_until_done(batch.id, poll_interval_seconds, max_wait_seconds)

        # batch.status は completed / failed / expired / cancelled のいずれか
        if completed.status in ("failed", "expired", "cancelled"):
            # output_file_id がない場合でも、せめて1件の失敗を返してユーザに通知する
            terminal_error = (
                f"batch ended with status={completed.status}"
                + (f": {completed.errors}" if getattr(completed, "errors", None) else "")
            )
            logger.error(f"[{self.model_config.id}] {terminal_error}")
            # 該当ジョブで output_file_id があれば部分回収を試みる
            results: list[BatchResult] = []
            if completed.output_file_id:
                results.extend(
                    self._parse_output(completed.output_file_id, output_structure)
                )
            if completed.error_file_id:
                results.extend(self._parse_errors(completed.error_file_id))
            if not results:
                # 全件失敗扱い（custom_id は不明なので入力件数分のフェイクは出さない）
                results.append(
                    BatchResult(custom_id="*", success=False, error=terminal_error)
                )
            return results

        results: list[BatchResult] = []
        if completed.output_file_id:
            results.extend(
                self._parse_output(completed.output_file_id, output_structure)
            )
        if completed.error_file_id:
            results.extend(self._parse_errors(completed.error_file_id))

        return results

    def _build_input_jsonl(
        self,
        requests: list[BatchRequest],
        templates: PromptTemplates,
        output_structure: type[BaseModel],
    ) -> bytes:
        """OpenAI Batch 用の JSONL を構築."""
        system_text = render_system(templates)
        schema = output_structure.model_json_schema()
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": output_structure.__name__,
                "schema": schema,
                "strict": True,
            },
        }

        lines: list[bytes] = []
        for req in requests:
            user_text = render_user(
                templates,
                req.character_info,
                req.criteria_description,
                req.log_json,
                criterion_name=req.criterion_name,
            )
            body = {
                "model": self.model_config.model,
                "messages": [
                    {"role": self.model_config.system_role, "content": system_text},
                    {"role": "user", "content": user_text},
                ],
                "response_format": response_format,
            }
            entry = {
                "custom_id": req.custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
            lines.append(json.dumps(entry, ensure_ascii=False).encode("utf-8"))

        return b"\n".join(lines) + b"\n"

    def _poll_until_done(
        self, batch_id: str, poll_interval: float, max_wait: float
    ):
        deadline = time.time() + max_wait
        last_status = None
        while True:
            batch = self._client.batches.retrieve(batch_id)
            if batch.status != last_status:
                logger.info(
                    f"[{self.model_config.id}] batch {batch_id} status: {batch.status}"
                )
                last_status = batch.status
            if batch.status in ("completed", "failed", "expired", "cancelled"):
                return batch
            if time.time() > deadline:
                raise TimeoutError(
                    f"[{self.model_config.id}] batch {batch_id} did not complete "
                    f"within {max_wait}s (status={batch.status})"
                )
            time.sleep(poll_interval)

    # 1ファイルあたり最大 100MB まで読み込む（通常のバッチ結果は十分に収まる）。
    MAX_FILE_BYTES = 100 * 1024 * 1024

    def _safe_read_file(self, file_id: str) -> str:
        """Files API の content をサイズ上限付きで安全に読み込む."""
        content = self._client.files.content(file_id)
        if hasattr(content, "read"):
            try:
                data = content.read(self.MAX_FILE_BYTES)
            except TypeError:
                # 古い openai SDK では .read() が引数を取らない
                data = content.read()
            if isinstance(data, bytes):
                if len(data) >= self.MAX_FILE_BYTES:
                    logger.warning(
                        f"[{self.model_config.id}] file {file_id} truncated at "
                        f"{self.MAX_FILE_BYTES} bytes"
                    )
                return data.decode("utf-8", errors="replace")
            return str(data)
        return str(content)

    def _parse_output(
        self, output_file_id: str, output_structure: type[BaseModel]
    ) -> list[BatchResult]:
        text = self._safe_read_file(output_file_id)

        results: list[BatchResult] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            custom_id = entry.get("custom_id", "?")
            response = entry.get("response") or {}
            error = entry.get("error")
            if error:
                results.append(
                    BatchResult(
                        custom_id=custom_id,
                        success=False,
                        error=json.dumps(error, ensure_ascii=False),
                    )
                )
                continue
            body = response.get("body") or {}
            try:
                content_str = body["choices"][0]["message"]["content"]
                parsed = output_structure.model_validate_json(content_str)
                results.append(BatchResult(custom_id=custom_id, success=True, response=parsed))
            except Exception as e:
                results.append(
                    BatchResult(custom_id=custom_id, success=False, error=str(e))
                )
        return results

    def _parse_errors(self, error_file_id: str) -> list[BatchResult]:
        text = self._safe_read_file(error_file_id)
        results: list[BatchResult] = []
        for line in text.splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            results.append(
                BatchResult(
                    custom_id=entry.get("custom_id", "?"),
                    success=False,
                    error=json.dumps(entry.get("error") or entry, ensure_ascii=False),
                )
            )
        return results
