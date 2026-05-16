"""RunDirectory のテスト."""

import json
from pathlib import Path

from src.aiwolf_log.game_log import AIWolfGameLog
from src.llm.client import ModelConfig
from src.processor.run_directory import (
    RUN_METADATA_FILENAME,
    RunDirectory,
    RunMetadata,
    load_metadata,
)


class _FakeGameLog:
    def __init__(self, game_id: str) -> None:
        self.game_id = game_id


def _model(id_: str) -> ModelConfig:
    return ModelConfig(id=id_, provider="openai", model="gpt-4o", api_key_env="K")


class TestRunDirectory:
    def test_create_makes_timestamped_dir(self, tmp_path: Path) -> None:
        rd = RunDirectory.create(tmp_path)
        assert rd.path.parent == tmp_path
        assert rd.path.is_dir()
        # timestamp format: YYYY-MM-DD_HH-MM-SS
        assert len(rd.timestamp) == 19

    def test_model_dir_creates_subdir(self, tmp_path: Path) -> None:
        rd = RunDirectory.create(tmp_path)
        d = rd.model_dir("gpt-4o")
        assert d == rd.path / "gpt-4o"
        assert d.is_dir()

    def test_latest_under_picks_most_recent(self, tmp_path: Path) -> None:
        (tmp_path / "2026-01-01_00-00-00").mkdir()
        (tmp_path / "2026-05-16_12-00-00").mkdir()
        (tmp_path / "2025-12-31_23-59-59").mkdir()
        latest = RunDirectory.latest_under(tmp_path)
        assert latest is not None
        assert latest.timestamp == "2026-05-16_12-00-00"

    def test_latest_under_empty_returns_none(self, tmp_path: Path) -> None:
        assert RunDirectory.latest_under(tmp_path) is None

    def test_write_and_load_metadata_roundtrip(self, tmp_path: Path) -> None:
        rd = RunDirectory.create(tmp_path)
        rd.write_metadata(
            input_dir=Path("data/input"),
            game_logs=[_FakeGameLog("g1"), _FakeGameLog("g2")],
            configured_models=[_model("gpt"), _model("claude")],
            eligible_models=[_model("gpt")],
            executed_model_ids=["gpt"],
            parallel_models=False,
            dry_run=True,
            dry_run_strict=False,
            use_batch_api=True,
        )

        # raw json check
        with (rd.path / RUN_METADATA_FILENAME).open() as f:
            data = json.load(f)
        assert data["game_count"] == 2
        assert data["use_batch_api"] is True

        # typed load
        meta = load_metadata(rd.path)
        assert isinstance(meta, RunMetadata)
        assert meta.game_count == 2
        assert meta.game_ids == ["g1", "g2"]
        assert meta.executed_models == ["gpt"]

    def test_iter_model_dirs(self, tmp_path: Path) -> None:
        rd = RunDirectory.create(tmp_path)
        rd.model_dir("gpt-4o")
        rd.model_dir("claude")
        # 同階層の run_metadata.json はファイルなので含まれない
        (rd.path / "irrelevant.txt").write_text("ignore")
        names = sorted(p.name for p in rd.iter_model_dirs())
        assert names == ["claude", "gpt-4o"]
