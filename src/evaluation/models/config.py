from __future__ import annotations

from typing import TYPE_CHECKING

from src.evaluation.models.criteria import EvaluationCriteria

if TYPE_CHECKING:
    from src.game.models import GameInfo


class EvaluationConfig(list[EvaluationCriteria]):
    """評価設定を表すクラス（EvaluationCriteriaのリストを継承）."""

    def get_criteria_for_game(self, game_info: GameInfo) -> list[EvaluationCriteria]:
        """指定されたゲームに適用される評価基準を取得."""
        return [criteria for criteria in self if criteria.applies_to(game_info)]

    def get_criteria_by_name(
        self, criteria_name: str, game_info: GameInfo
    ) -> EvaluationCriteria:
        """基準名で評価基準を取得."""
        for criteria in self.get_criteria_for_game(game_info):
            if criteria.name == criteria_name:
                return criteria
        raise KeyError(
            f"Criteria '{criteria_name}' not found for game "
            f"(players={game_info.player_count}, werewolves={game_info.werewolf_count})"
        )
