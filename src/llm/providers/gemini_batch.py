"""Gemini Batch API クライアント.

provider: gemini    -> Google AI Studio。inline source（リクエストのリスト直接渡し）対応。
provider: vertex_ai -> Vertex AI。GCS の JSONL ファイル経由のみサポート。
                       ModelConfig.gcs_bucket（gs://bucket/prefix/）の指定が必須。
                       google-cloud-storage SDK を直接使う。

フロー（共通）:
    1. リクエスト群を JSONL/inline に整形
    2. client.batches.create(src=..., config=...) でジョブ作成
    3. client.batches.get(name=...) でポーリング
    4. ジョブ完了後、dest から結果を回収

24h SLA, 50% 割引。
"""

import json
import logging
import os
import time
import uuid

from google import genai
from google.genai import types
from pydantic import BaseModel

from src.llm.batch import BatchClient, BatchRequest, BatchResult
from src.llm.client import ModelConfig, PromptTemplates

logger = logging.getLogger(__name__)


_TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


class GeminiBatchClient(BatchClient):
    """Gemini Batch API クライアント（AI Studio + Vertex AI）."""

    AI_STUDIO_PROVIDER = "gemini"
    VERTEX_PROVIDER = "vertex_ai"

    def __init__(self, model_config: ModelConfig) -> None:
        self.model_config = model_config
        if model_config.provider == self.VERTEX_PROVIDER:
            if not model_config.project_id or not model_config.location:
                raise ValueError(
                    f"{model_config.id}: project_id and location are required for "
                    "vertex_ai provider"
                )
            if not model_config.gcs_bucket:
                raise ValueError(
                    f"{model_config.id}: gcs_bucket is required for vertex_ai batch "
                    "mode (e.g. 'gs://my-bucket/aiwolf-judge/')"
                )
            self._client = genai.Client(
                vertexai=True,
                project=model_config.project_id,
                location=model_config.location,
            )
            self._is_vertex = True
        else:
            if not model_config.api_key_env:
                raise ValueError(
                    f"{model_config.id}: api_key_env is required for gemini provider"
                )
            api_key = os.environ.get(model_config.api_key_env)
            if not api_key:
                raise ValueError(
                    f"{model_config.id}: env var {model_config.api_key_env} is not set"
                )
            self._client = genai.Client(api_key=api_key)
            self._is_vertex = False

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

        # custom_id -> リクエスト本体（dict）。順序保持のため OrderedDict 風に。
        request_dicts: list[dict] = []
        custom_ids: list[str] = []
        for req in requests:
            entry = self._build_request_entry(req, templates, output_structure)
            request_dicts.append(entry)
            custom_ids.append(req.custom_id)

        # Vertex AI 経路は GCS に input/output を置くため、終わったら必ず削除する
        gcs_artifacts: dict[str, str] | None = None
        try:
            if self._is_vertex:
                src, dest, gcs_artifacts = self._stage_to_gcs(request_dicts)
                batch_job = self._client.batches.create(
                    model=self.model_config.model,
                    src=src,
                    config=types.CreateBatchJobConfig(dest=dest),
                )
            else:
                # AI Studio: inline_requests を直接渡せる
                batch_job = self._client.batches.create(
                    model=self.model_config.model,
                    src=request_dicts,
                )

            job_name = batch_job.name
            logger.info(
                f"[{self.model_config.id}] batch created: name={job_name}, "
                f"state={getattr(batch_job, 'state', None)}"
            )

            completed = self._poll_until_done(
                job_name, poll_interval_seconds, max_wait_seconds
            )
            return self._collect_results(completed, custom_ids, output_structure)
        finally:
            if gcs_artifacts is not None:
                self._cleanup_gcs_artifacts(gcs_artifacts)

    # ------------------------------------------------------------------
    # Request building
    # ------------------------------------------------------------------

    def _build_request_entry(
        self,
        req: BatchRequest,
        templates: PromptTemplates,
        output_structure: type[BaseModel],
    ) -> dict:
        from jinja2 import Template

        system_text = Template(templates.system).render().strip()
        user_text = (
            Template(templates.user)
            .render(
                character_info=req.character_info,
                criteria_description=req.criteria_description,
                log=req.log_json,
            )
            .strip()
        )
        schema = output_structure.model_json_schema()
        return {
            "contents": [{"role": "user", "parts": [{"text": user_text}]}],
            "system_instruction": {"parts": [{"text": system_text}]},
            "generation_config": {
                "response_mime_type": "application/json",
                "response_schema": schema,
            },
        }

    # ------------------------------------------------------------------
    # Vertex AI: stage JSONL to GCS
    # ------------------------------------------------------------------

    def _stage_to_gcs(
        self, request_dicts: list[dict]
    ) -> tuple[str, types.BatchJobDestination, dict[str, str]]:
        """JSONL を GCS に置いて (src_uri, dest_config, cleanup_info) を返す.

        cleanup_info は submit_and_wait の finally でクリーンアップに渡される。
        """
        try:
            from google.cloud import storage  # noqa: F401  # for ImportError check
        except ImportError as e:
            raise RuntimeError(
                "google-cloud-storage is required for Vertex AI batch mode. "
                "Install with: uv add google-cloud-storage"
            ) from e

        bucket_uri = self.model_config.gcs_bucket
        if not bucket_uri.startswith("gs://"):
            raise ValueError(
                f"gcs_bucket must start with 'gs://' (got: {bucket_uri})"
            )
        path = bucket_uri[len("gs://"):]
        parts = path.split("/", 1)
        bucket_name = parts[0]
        prefix = parts[1].rstrip("/") if len(parts) > 1 else ""

        run_id = uuid.uuid4().hex[:12]
        input_blob_name = (
            f"{prefix}/{self.model_config.id}/{run_id}/input.jsonl"
            if prefix
            else f"{self.model_config.id}/{run_id}/input.jsonl"
        )
        output_prefix_name = (
            f"{prefix}/{self.model_config.id}/{run_id}/output/"
            if prefix
            else f"{self.model_config.id}/{run_id}/output/"
        )

        from google.cloud import storage as gcs_storage

        client = gcs_storage.Client(project=self.model_config.project_id)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(input_blob_name)
        jsonl_payload = "\n".join(
            json.dumps({"request": entry}, ensure_ascii=False) for entry in request_dicts
        )
        blob.upload_from_string(jsonl_payload, content_type="application/jsonl")

        src_uri = f"gs://{bucket_name}/{input_blob_name}"
        dest_uri = f"gs://{bucket_name}/{output_prefix_name}"
        logger.info(f"[{self.model_config.id}] uploaded batch input to {src_uri}")

        return (
            src_uri,
            types.BatchJobDestination(format="jsonl", gcs_uri=dest_uri),
            {
                "bucket_name": bucket_name,
                "input_blob_name": input_blob_name,
                "output_prefix": output_prefix_name,
            },
        )

    def _cleanup_gcs_artifacts(self, artifacts: dict[str, str]) -> None:
        """バッチで使った GCS 入力 blob と出力 prefix を削除.

        失敗しても処理は継続（ログのみ）。ユーザーがリージョン/権限を絞っている場合に
        備えてベストエフォート。
        """
        try:
            from google.cloud import storage
        except ImportError:
            return

        bucket_name = artifacts.get("bucket_name")
        if not bucket_name:
            return

        try:
            client = storage.Client(project=self.model_config.project_id)
            bucket = client.bucket(bucket_name)
        except Exception as e:
            logger.warning(
                f"[{self.model_config.id}] GCS cleanup skipped (client init failed): {e}"
            )
            return

        # 入力 blob の削除
        input_blob_name = artifacts.get("input_blob_name")
        if input_blob_name:
            try:
                bucket.blob(input_blob_name).delete()
                logger.info(
                    f"[{self.model_config.id}] cleaned up GCS input "
                    f"gs://{bucket_name}/{input_blob_name}"
                )
            except Exception as e:
                logger.warning(
                    f"[{self.model_config.id}] failed to delete input blob "
                    f"{input_blob_name}: {e}"
                )

        # 出力 prefix 配下の全 blob 削除
        output_prefix = artifacts.get("output_prefix")
        if output_prefix:
            try:
                deleted = 0
                for blob in bucket.list_blobs(prefix=output_prefix):
                    blob.delete()
                    deleted += 1
                logger.info(
                    f"[{self.model_config.id}] cleaned up {deleted} output blob(s) "
                    f"under gs://{bucket_name}/{output_prefix}"
                )
            except Exception as e:
                logger.warning(
                    f"[{self.model_config.id}] failed to clean output prefix "
                    f"{output_prefix}: {e}"
                )

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    def _poll_until_done(self, job_name: str, poll_interval: float, max_wait: float):
        deadline = time.time() + max_wait
        last_state = None
        while True:
            job = self._client.batches.get(name=job_name)
            state = str(getattr(job, "state", "UNKNOWN"))
            if state != last_state:
                logger.info(
                    f"[{self.model_config.id}] batch {job_name} state: {state}"
                )
                last_state = state
            if state in _TERMINAL_STATES or any(s in state for s in _TERMINAL_STATES):
                return job
            if time.time() > deadline:
                raise TimeoutError(
                    f"[{self.model_config.id}] batch {job_name} did not finish "
                    f"within {max_wait}s (state={state})"
                )
            time.sleep(poll_interval)

    # ------------------------------------------------------------------
    # Result collection
    # ------------------------------------------------------------------

    def _collect_results(
        self,
        job,
        custom_ids: list[str],
        output_structure: type[BaseModel],
    ) -> list[BatchResult]:
        state = str(getattr(job, "state", ""))
        if "SUCCEEDED" not in state:
            error = getattr(job, "error", None)
            return [
                BatchResult(
                    custom_id=cid,
                    success=False,
                    error=f"batch ended with state={state}, error={error}",
                )
                for cid in custom_ids
            ]

        dest = getattr(job, "dest", None)
        if self._is_vertex:
            return self._collect_results_from_gcs(dest, custom_ids, output_structure)
        return self._collect_results_inline(dest, custom_ids, output_structure)

    def _collect_results_inline(
        self,
        dest,
        custom_ids: list[str],
        output_structure: type[BaseModel],
    ) -> list[BatchResult]:
        inlined = getattr(dest, "inlined_responses", None) if dest else None
        if not inlined:
            return [
                BatchResult(
                    custom_id=cid,
                    success=False,
                    error="batch succeeded but no inlined_responses",
                )
                for cid in custom_ids
            ]

        results: list[BatchResult] = []
        for cid, resp in zip(custom_ids, inlined):
            result = self._parse_inlined_response(cid, resp, output_structure)
            results.append(result)
        return results

    def _parse_inlined_response(
        self,
        custom_id: str,
        resp,
        output_structure: type[BaseModel],
    ) -> BatchResult:
        err = getattr(resp, "error", None)
        if err:
            return BatchResult(custom_id=custom_id, success=False, error=str(err))
        response_obj = getattr(resp, "response", None)
        if response_obj is None:
            return BatchResult(custom_id=custom_id, success=False, error="no response")
        text = getattr(response_obj, "text", None)
        if not text:
            candidates = getattr(response_obj, "candidates", None) or []
            if candidates:
                content = getattr(candidates[0], "content", None)
                parts = getattr(content, "parts", None) or [] if content else []
                text = "".join(getattr(p, "text", "") or "" for p in parts)
        if not text:
            return BatchResult(
                custom_id=custom_id, success=False, error="empty response text"
            )
        try:
            parsed = output_structure.model_validate_json(text)
            return BatchResult(custom_id=custom_id, success=True, response=parsed)
        except Exception as e:
            return BatchResult(custom_id=custom_id, success=False, error=str(e))

    def _collect_results_from_gcs(
        self,
        dest,
        custom_ids: list[str],
        output_structure: type[BaseModel],
    ) -> list[BatchResult]:
        try:
            from google.cloud import storage
        except ImportError as e:
            raise RuntimeError("google-cloud-storage required for Vertex batch") from e

        gcs_uri = getattr(dest, "gcs_uri", None) if dest else None
        if not gcs_uri:
            return [
                BatchResult(
                    custom_id=cid,
                    success=False,
                    error="batch succeeded but no GCS destination",
                )
                for cid in custom_ids
            ]

        path = gcs_uri[len("gs://"):]
        bucket_name, _, prefix = path.partition("/")
        client = storage.Client(project=self.model_config.project_id)
        bucket = client.bucket(bucket_name)
        blobs = list(bucket.list_blobs(prefix=prefix))
        if not blobs:
            return [
                BatchResult(
                    custom_id=cid, success=False, error=f"no output blobs in {gcs_uri}"
                )
                for cid in custom_ids
            ]

        # 入力順と出力順は同じであることが期待される
        text_chunks: list[str] = []
        for blob in blobs:
            if not blob.name.endswith(".jsonl"):
                continue
            text_chunks.append(blob.download_as_text())
        text = "\n".join(text_chunks)

        results: list[BatchResult] = []
        lines = [line for line in text.splitlines() if line.strip()]
        for cid, line in zip(custom_ids, lines):
            try:
                entry = json.loads(line)
                response_obj = entry.get("response") or {}
                candidates = response_obj.get("candidates") or []
                if not candidates:
                    raise ValueError("no candidates")
                content = candidates[0].get("content") or {}
                parts = content.get("parts") or []
                text_payload = "".join(p.get("text", "") for p in parts)
                if not text_payload:
                    raise ValueError("empty response text")
                parsed = output_structure.model_validate_json(text_payload)
                results.append(BatchResult(custom_id=cid, success=True, response=parsed))
            except Exception as e:
                results.append(
                    BatchResult(custom_id=cid, success=False, error=str(e))
                )

        # 不足分は失敗として埋める
        for cid in custom_ids[len(results):]:
            results.append(
                BatchResult(custom_id=cid, success=False, error="missing output line")
            )
        return results
