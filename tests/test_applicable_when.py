"""ApplicableWhen の条件判定テスト."""

import pytest

from src.evaluation.models.criteria import ApplicableWhen, EvaluationCriteria, RankingType
from src.game.models import GameFormat, GameInfo


def _make_game(*, players: int, werewolves: int) -> GameInfo:
    return GameInfo(
        game_format=GameFormat.MAIN_MATCH,
        player_count=players,
        werewolf_count=werewolves,
        game_id="g",
    )


class TestApplicableWhen:
    def test_no_condition_always_applies(self) -> None:
        criteria = EvaluationCriteria(
            name="x",
            description="d",
            ranking_type=RankingType.ORDINAL,
            order=1,
            applicable_when=None,
        )
        assert criteria.applies_to(_make_game(players=5, werewolves=1))
        assert criteria.applies_to(_make_game(players=13, werewolves=3))

    @pytest.mark.parametrize(
        "werewolf_count,expected",
        [(0, False), (1, False), (2, True), (3, True), (10, True)],
    )
    def test_werewolf_count_gte(self, werewolf_count: int, expected: bool) -> None:
        criteria = EvaluationCriteria(
            name="team_play",
            description="d",
            ranking_type=RankingType.ORDINAL,
            order=6,
            applicable_when=ApplicableWhen(werewolf_count_gte=2),
        )
        game = _make_game(players=13, werewolves=werewolf_count)
        assert criteria.applies_to(game) is expected
