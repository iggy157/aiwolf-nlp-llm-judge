"""AIWolfログをLLM用のJSONL形式に変換するモジュール."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from src.aiwolf_log.parser import AIWolfCSVParser
from src.game.models import GameFormat

if TYPE_CHECKING:
    from src.aiwolf_log.game_log import AIWolfGameLog

logger = logging.getLogger(__name__)


class GameLogFormatter:
    """ゲームログをLLM用のJSONL形式に変換するクラス."""

    def __init__(
        self,
        game_log: AIWolfGameLog,
        config: dict[str, Any] | None = None,
        parser: AIWolfCSVParser | None = None,
    ) -> None:
        """初期化.

        Args:
            game_log: AIWolfGameLogインスタンス
            config: CSVリーダー用の設定
            parser: AIWolfCSVParserインスタンス
        """
        self.game_log = game_log
        self.config = config or {}
        self.parser = parser or AIWolfCSVParser()
        self._player_mapping: dict[str, str] | None = None

    @property
    def player_mapping(self) -> dict[str, str]:
        """プレイヤーインデックスから名前へのマッピング（遅延初期化）."""
        if self._player_mapping is None:
            self._player_mapping = self._create_player_mapping()
        return self._player_mapping

    def convert_to_jsonl(
        self,
        game_format: GameFormat | None = None,
        exclude_actions: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """ゲームログをJSONL形式のデータに変換する.

        Args:
            game_format: ゲーム形式
            exclude_actions: 出力から除外するアクション名の集合（例: {"result"} で勝敗を隠す）。
                line_number は除外前のログ上の位置をそのまま保持する。

        Returns:
            各行がdictとして格納されたリスト

        Raises:
            ValueError: ログの読み込みまたは変換に失敗した場合
        """
        jsonl_data: list[dict[str, Any]] = []
        line_number = 0
        excluded = exclude_actions or set()

        try:
            with self.game_log.get_csv_reader(self.config) as reader:
                while line := reader.read_next_line():
                    formatted_line = self._process_line(line, line_number, game_format)
                    line_number += 1
                    if formatted_line.get("action") in excluded:
                        continue
                    jsonl_data.append(formatted_line)

        except Exception as e:
            raise ValueError(
                f"Failed to convert game log {self.game_log.game_id}: {e}"
            ) from e

        return jsonl_data

    def _process_line(
        self, line: list[str], line_number: int, game_format: GameFormat | None
    ) -> dict[str, Any]:
        """単一行を処理してフォーマット済みデータを返す."""
        formatted_line = self.parser.parse_action_data(line)
        formatted_line["line_number"] = line_number

        # team_nameの末尾数字除去（main_matchの場合）
        self._normalize_team_name(formatted_line, game_format)

        # インデックスキーをプレイヤー名に変換（statusアクション以外）
        if formatted_line.get("action") != "status":
            formatted_line = self._convert_index_keys(formatted_line)

        return formatted_line

    def _normalize_team_name(
        self, data: dict[str, Any], game_format: GameFormat | None
    ) -> None:
        """team_nameの正規化を行う（インプレース）."""
        if (
            game_format == GameFormat.MAIN_MATCH
            and "team_name" in data
            and data["team_name"]
        ):
            data["team_name"] = self._remove_trailing_digits(data["team_name"])

    def _convert_index_keys(self, data: dict[str, Any]) -> dict[str, Any]:
        """*_indexキーをプレイヤー名に変換する."""
        if not self.player_mapping:
            return data

        converted_data = {}
        index_pattern = re.compile(r"(.+)_index$")

        for key, value in data.items():
            if match := index_pattern.match(key):
                base_key = match.group(1)
                player_name = self.player_mapping.get(str(value))
                if player_name:
                    converted_data[base_key] = player_name
                else:
                    # マッピングが見つからない場合は元の値を保持
                    converted_data[key] = value
            else:
                converted_data[key] = value

        return converted_data

    def _create_player_mapping(self) -> dict[str, str]:
        """ログのstatusエントリからプレイヤーインデックスから名前へのマッピングを作成する."""
        mapping: dict[str, str] = {}

        try:
            with self.game_log.get_csv_reader(self.config) as reader:
                while line := reader.read_next_line():
                    parsed_data = self.parser.parse_action_data(line)

                    # statusアクションからplayer_indexとplayer_nameを取得
                    if (
                        parsed_data.get("action") == "status"
                        and "player_index" in parsed_data
                        and "player_name" in parsed_data
                    ):
                        player_index = parsed_data["player_index"]
                        player_name = parsed_data["player_name"]

                        if player_index and player_name:
                            mapping[player_index] = player_name

        except FileNotFoundError as e:
            logger.warning(f"Log file not found: {e}")
        except PermissionError as e:
            logger.error(f"Permission denied reading log file: {e}")
        except Exception as e:
            logger.error(f"Unexpected error reading log file: {e}")

        return mapping

    @staticmethod
    def _remove_trailing_digits(text: str) -> str:
        """文字列の末尾の数字を除去する."""
        return re.sub(r"\d+$", "", text)
