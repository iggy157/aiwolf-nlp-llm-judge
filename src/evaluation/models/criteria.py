from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # 循環インポート回避: GameInfo は型注釈でしか参照しない（実行時は不要）。
    # src.game.detector -> src.evaluation.loaders -> src.evaluation.models と
    # チェーンが繋がるため、ここで直接 import するとループする。
    from src.game.models import GameInfo


class RankingType(Enum):
    """ランキングの型を表す列挙型."""

    ORDINAL = "ordinal"  # 順序付け（1位、2位...）
    COMPARATIVE = "comparative"  # 比較ベース（A > B > C）


@dataclass
class ApplicableWhen:
    """評価基準の適用条件を表すデータクラス.

    各フィールドは None ならその条件を課さない。
    複数フィールドが指定された場合は AND で結合される。
    """

    werewolf_count_gte: int | None = None

    def matches(self, game_info: GameInfo) -> bool:
        """指定されたゲーム情報が条件を満たすかを判定."""
        if self.werewolf_count_gte is not None:
            if game_info.werewolf_count < self.werewolf_count_gte:
                return False
        return True


@dataclass
class EvaluationCriteria:
    """評価基準を表すデータクラス."""

    name: str
    description: str
    ranking_type: RankingType
    order: int = 0
    applicable_when: ApplicableWhen | None = None

    def applies_to(self, game_info: GameInfo) -> bool:
        """この評価基準が指定されたゲームに適用されるかを判定."""
        if self.applicable_when is None:
            return True
        return self.applicable_when.matches(game_info)
