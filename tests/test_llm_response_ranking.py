"""EvaluationLLMResponse の競技ランキング(1224)方式 validator のテスト."""

import pytest

from src.evaluation.models.llm_response import (
    EvaluationElement,
    EvaluationLLMResponse,
)


def _make(ranks: list[int]) -> EvaluationLLMResponse:
    """順位列から EvaluationLLMResponse を構築（validator 起動用）."""
    rankings = [
        EvaluationElement(
            player_name=f"P{i + 1}",
            reasoning="r",
            ranking=r,
        )
        for i, r in enumerate(ranks)
    ]
    return EvaluationLLMResponse(rankings=rankings)


class TestValidRankings:
    """1224式として valid な順位列."""

    def test_strict_total_order(self):
        _make([1, 2, 3, 4, 5])

    def test_top_three_tied(self):
        _make([1, 1, 1, 4, 5])

    def test_middle_pair_tied(self):
        _make([1, 2, 2, 4, 5])

    def test_bottom_pair_tied(self):
        _make([1, 2, 3, 4, 4])

    def test_two_disjoint_ties(self):
        _make([1, 1, 3, 4, 4])

    def test_bottom_triple_tied(self):
        _make([1, 2, 3, 3, 3])

    def test_two_only(self):
        _make([1, 2])

    def test_two_distinct_ranks_minimum(self):
        # 全員同順位は禁止だが、最低2階位あれば OK
        _make([1, 1, 1, 1, 5])


class TestInvalidRankings:
    """validation error を期待するケース."""

    def test_dense_rank_style_rejected(self):
        # タイの分を詰めてしまう dense rank 方式は invalid
        with pytest.raises(ValueError, match="競技ランキング方式と整合"):
            _make([1, 1, 1, 2, 3])

    def test_all_same_rank_rejected(self):
        with pytest.raises(ValueError, match="全員を同順位"):
            _make([1, 1, 1, 1, 1])

    def test_top_not_one(self):
        with pytest.raises(ValueError, match="最上位の順位は1"):
            _make([2, 3, 4, 5, 6])

    def test_max_exceeds_n(self):
        with pytest.raises(ValueError):
            _make([1, 2, 3, 4, 6])

    def test_empty_rankings(self):
        with pytest.raises(ValueError, match="空"):
            EvaluationLLMResponse(rankings=[])

    def test_partial_skip_inconsistent(self):
        # 2,2 のあとは 4 に飛ぶべき。3 だと invalid
        with pytest.raises(ValueError, match="競技ランキング方式と整合"):
            _make([1, 2, 2, 3, 5])
