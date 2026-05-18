"""1ゲーム分の評価前処理を集約するビルダー.

`GameProcessor`（同期パス）と `BatchOrchestrator`（バッチパス）の両方で
「game_info 検出 + 評価基準絞り込み + ログ整形 + character_info + agent→team」を
重複実装していたものを統合する。
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.aiwolf_log.game_log import AIWolfGameLog
from src.evaluation.models.config import EvaluationConfig
from src.evaluation.models.criteria import EvaluationCriteria
from src.game.detector import GameDetector
from src.game.models import GameInfo
from src.llm.formatter import GameLogFormatter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GameContext:
    """1ゲーム分の評価に必要な前処理結果."""

    game_log: AIWolfGameLog
    game_info: GameInfo
    criteria: list[EvaluationCriteria]
    formatted_data: list[dict[str, Any]]
    character_info: str
    agent_to_team_mapping: dict[str, str]


class GameContextBuilder:
    """`GameContext` を生成するサービス.

    同期 / バッチ両方のパスから呼ばれる前処理ロジックを一か所に集約。
    """

    def __init__(
        self,
        settings_path: Path,
        evaluation_config: EvaluationConfig,
        reader_config: dict[str, Any] | None = None,
    ) -> None:
        """初期化.

        Args:
            settings_path: settings.yaml のパス（GameDetector へ）
            evaluation_config: 評価基準セット
            reader_config: ログCSV読み込み時の追加設定（encoding 等）
        """
        self.settings_path = settings_path
        self.evaluation_config = evaluation_config
        self.reader_config = reader_config or {}

    def build(self, game_log: AIWolfGameLog) -> GameContext:
        """指定ゲームの GameContext を構築."""
        game_info = GameDetector.detect_game_format(
            game_log.log_path, self.settings_path
        )
        criteria = self.evaluation_config.get_criteria_for_game(game_info)

        formatter = GameLogFormatter(game_log, self.reader_config)
        # 評価バイアス対策: 勝敗(result アクション)はログから除外して LLM に見せない。
        # 投票結果・占い結果・処刑などは翌日以降の議題になり得るため残す。
        formatted_data = formatter.convert_to_jsonl(
            game_info.game_format,
            exclude_actions={"result"},
        )

        character_info = self._build_character_info(game_log)
        agent_to_team_mapping = game_log.get_agent_to_team_mapping()

        return GameContext(
            game_log=game_log,
            game_info=game_info,
            criteria=criteria,
            formatted_data=formatted_data,
            character_info=character_info,
            agent_to_team_mapping=agent_to_team_mapping,
        )

    @staticmethod
    def _build_character_info(game_log: AIWolfGameLog) -> str:
        """JSON 側のプロフィールから "- 名前: プロフィール" 形式の文字列を生成."""
        try:
            profiles = game_log.get_character_info()
        except Exception as e:
            logger.warning(
                f"Failed to load character info for {game_log.game_id}: {e}"
            )
            return ""
        return "\n".join(f"- {name}: {profile}" for name, profile in profiles.items())
