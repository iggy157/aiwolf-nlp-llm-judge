"""実行毎のタイムスタンプディレクトリと run_metadata.json を管理するヘルパ.

`<output_root>/<timestamp>/<model_id>/...` の階層を司る単一の責務クラス。
作成・メタデータ書き出し・最新タイムスタンプの探索を担当する。
"""

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from src.aiwolf_log.game_log import AIWolfGameLog
from src.llm.client import ModelConfig

logger = logging.getLogger(__name__)

RUN_DIR_TS_FORMAT = "%Y-%m-%d_%H-%M-%S"
RUN_METADATA_FILENAME = "run_metadata.json"


class RunDirectory:
    """実行毎のタイムスタンプディレクトリを管理する値オブジェクト的クラス."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @property
    def timestamp(self) -> str:
        return self.path.name

    def model_dir(self, model_id: str) -> Path:
        """指定モデルの出力ディレクトリを返す（必要なら作成）."""
        d = self.path / model_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    @classmethod
    def create(cls, output_root: Path) -> "RunDirectory":
        """新しいタイムスタンプディレクトリを作成して返す."""
        timestamp = datetime.now().strftime(RUN_DIR_TS_FORMAT)
        path = output_root / timestamp
        path.mkdir(parents=True, exist_ok=True)
        return cls(path)

    @classmethod
    def latest_under(cls, output_root: Path) -> "RunDirectory | None":
        """`output_root` 直下で最も新しいタイムスタンプディレクトリを返す."""
        if not output_root.is_dir():
            return None
        timestamp_dirs = sorted(
            (p for p in output_root.iterdir() if p.is_dir()),
            key=lambda p: p.name,
        )
        return cls(timestamp_dirs[-1]) if timestamp_dirs else None

    def write_metadata(
        self,
        *,
        input_dir: Path,
        game_logs: list[AIWolfGameLog],
        configured_models: list[ModelConfig],
        eligible_models: list[ModelConfig],
        executed_model_ids: list[str],
        parallel_models: bool,
        dry_run: bool,
        dry_run_strict: bool,
        use_batch_api: bool = False,
    ) -> None:
        """run_metadata.json を書き出す."""
        metadata = {
            "run_timestamp": self.timestamp,
            "input_dir": str(input_dir),
            "game_count": len(game_logs),
            "game_ids": [gl.game_id for gl in game_logs],
            "models": [asdict(m) for m in configured_models],
            "eligible_models": [m.id for m in eligible_models],
            "executed_models": executed_model_ids,
            "parallel_models": parallel_models,
            "dry_run": dry_run,
            "dry_run_strict": dry_run_strict,
            "use_batch_api": use_batch_api,
        }
        metadata_path = self.path / RUN_METADATA_FILENAME
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"Run metadata saved: {metadata_path}")

    def iter_model_dirs(self) -> list[Path]:
        """ディレクトリ配下のモデル別サブディレクトリ一覧を返す."""
        return sorted(p for p in self.path.iterdir() if p.is_dir())

    def __fspath__(self) -> str:
        return str(self.path)

    def __str__(self) -> str:
        return str(self.path)


def load_metadata(run_dir: Path) -> dict[str, Any] | None:
    """run_metadata.json を読み込む（存在しない場合は None）."""
    metadata_path = run_dir / RUN_METADATA_FILENAME
    if not metadata_path.is_file():
        return None
    try:
        with metadata_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to load {metadata_path}: {e}")
        return None
