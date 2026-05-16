"""GameDetector の player_count / werewolf_count 自動検出テスト."""

import csv
from pathlib import Path

import pytest
import yaml

from src.game.detector import GameDetector


@pytest.fixture
def settings_yaml(tmp_path: Path) -> Path:
    settings = {
        "path": {"env": "x.env", "evaluation_criteria": "x.yaml"},
        "game": {"format": "main_match"},
        "processing": {"input_dir": "/x", "output_dir": "/y"},
        "llm": {
            "prompt_yml": "x.yaml",
            "models": [
                {"id": "m", "provider": "openai", "model": "gpt-4o", "api_key_env": "K"}
            ],
        },
    }
    p = tmp_path / "settings.yaml"
    p.write_text(yaml.safe_dump(settings))
    return p


def _write_log(path: Path, roles: list[str]) -> None:
    """指定 roles に対応する day 0 status 行を持つログ CSV を書き出す."""
    rows = []
    for i, role in enumerate(roles, 1):
        rows.append(["0", "status", str(i), role, "ALIVE", f"team_{i}", f"P{i}"])
    # ノイズ行を1つ
    rows.append(["0", "talk", "1", "1", "1", "hello"])
    with path.open("w", newline="") as f:
        csv.writer(f).writerows(rows)


class TestGameDetector:
    def test_5_player_1_werewolf(self, tmp_path: Path, settings_yaml: Path) -> None:
        log = tmp_path / "g.log"
        _write_log(log, ["VILLAGER", "VILLAGER", "SEER", "POSSESSED", "WEREWOLF"])
        info = GameDetector.detect_game_format(log, settings_yaml)
        assert info.player_count == 5
        assert info.werewolf_count == 1

    def test_13_player_3_werewolf(self, tmp_path: Path, settings_yaml: Path) -> None:
        roles = ["VILLAGER"] * 6 + [
            "SEER",
            "MEDIUM",
            "BODYGUARD",
            "POSSESSED",
            "WEREWOLF",
            "WEREWOLF",
            "WEREWOLF",
        ]
        log = tmp_path / "g13.log"
        _write_log(log, roles)
        info = GameDetector.detect_game_format(log, settings_yaml)
        assert info.player_count == 13
        assert info.werewolf_count == 3

    def test_9_player_2_werewolf(self, tmp_path: Path, settings_yaml: Path) -> None:
        """9人戦（人狼2人）— team_play 評価が適用されるべきケース."""
        roles = ["VILLAGER"] * 4 + [
            "SEER",
            "MEDIUM",
            "POSSESSED",
            "WEREWOLF",
            "WEREWOLF",
        ]
        log = tmp_path / "g9.log"
        _write_log(log, roles)
        info = GameDetector.detect_game_format(log, settings_yaml)
        assert info.player_count == 9
        assert info.werewolf_count == 2

    def test_missing_log_raises(self, tmp_path: Path, settings_yaml: Path) -> None:
        with pytest.raises(FileNotFoundError):
            GameDetector.detect_game_format(tmp_path / "nope.log", settings_yaml)

    def test_no_status_rows_raises(self, tmp_path: Path, settings_yaml: Path) -> None:
        log = tmp_path / "empty.log"
        # status 行が全くないログ
        with log.open("w", newline="") as f:
            csv.writer(f).writerows([["0", "talk", "1", "1", "1", "hi"]])
        with pytest.raises(ValueError, match="No initial.*status"):
            GameDetector.detect_game_format(log, settings_yaml)
