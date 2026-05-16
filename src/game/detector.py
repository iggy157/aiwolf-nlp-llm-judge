import csv
from pathlib import Path

from src.aiwolf_log.csv_schema import ActionTypes, CSVColumnIndices
from src.evaluation.loaders.settings_loader import SettingsLoader
from src.game.models import GameInfo


WEREWOLF_ROLE = "WEREWOLF"


class GameDetector:
    """CSVファイルからゲーム形式を検出するクラス."""

    @staticmethod
    def detect_game_format(csv_path: Path, settings_path: Path) -> GameInfo:
        """CSVファイルからゲーム形式を検出.

        プレイヤー数と人狼の人数はDay 0のstatus行から自動検出する。
        ゲーム形式（main_match/self_match）はログから判別できないため
        settings.yamlから読み込む。

        Args:
            csv_path: CSVファイルのパス
            settings_path: 設定ファイルのパス

        Returns:
            GameInfo: 検出されたゲーム情報

        Raises:
            FileNotFoundError: CSVファイルが見つからない場合
            ValueError: ゲーム形式を判定できない場合
        """
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        game_format = SettingsLoader.load_game_format(settings_path)
        player_count, werewolf_count = GameDetector._count_roles_from_initial_status(
            csv_path
        )

        return GameInfo(
            game_format=game_format,
            player_count=player_count,
            werewolf_count=werewolf_count,
            game_id=csv_path.stem,
        )

    @staticmethod
    def _count_roles_from_initial_status(csv_path: Path) -> tuple[int, int]:
        """Day 0のstatus行を集計してプレイヤー数と人狼数を返す.

        Args:
            csv_path: CSVファイルのパス

        Returns:
            (player_count, werewolf_count)

        Raises:
            ValueError: status行が見つからない場合、もしくは読み込みに失敗した場合
        """
        player_count = 0
        werewolf_count = 0

        try:
            with csv_path.open("r", encoding="utf-8") as f:
                for row in csv.reader(f):
                    if len(row) <= CSVColumnIndices.ACTION:
                        continue
                    if row[CSVColumnIndices.DAY] != "0":
                        continue
                    if row[CSVColumnIndices.ACTION].lower() != ActionTypes.STATUS:
                        continue

                    player_count += 1
                    role_index = CSVColumnIndices.StatusAction.ROLE
                    if len(row) > role_index and row[role_index] == WEREWOLF_ROLE:
                        werewolf_count += 1
        except OSError as e:
            raise ValueError(f"Failed to read CSV file: {e}") from e

        if player_count == 0:
            raise ValueError(
                f"No initial (day 0) status rows found in {csv_path}; "
                "cannot determine player count"
            )

        return player_count, werewolf_count
