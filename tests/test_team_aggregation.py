"""TeamAggregationService と EvaluationResult.from_dict のテスト."""

from src.evaluation.models.result import EvaluationResult
from src.processor.pipeline.team_aggregation import TeamAggregationService


def _mappings() -> dict:
    return {
        "criteria_name_to_description": {
            "natural_expression": "発話表現は自然か",
            "team_play": "チームプレイができているか",
        },
        "criteria_name_to_order": {
            "natural_expression": 1,
            "team_play": 6,
        },
    }


def _sample_eval_dict() -> dict:
    return {
        "natural_expression": {
            "rankings": [
                {
                    "player_name": "Alice",
                    "team": "team_a",
                    "ranking": 1,
                    "reasoning": "r1",
                },
                {
                    "player_name": "Bob",
                    "team": "team_b",
                    "ranking": 2,
                    "reasoning": "r2",
                },
            ]
        }
    }


class TestEvaluationResultFromDict:
    def test_from_flat_dict(self) -> None:
        result = EvaluationResult.from_dict(_sample_eval_dict())
        assert result.get_criteria_names() == ["natural_expression"]
        criterion = result[0]
        assert len(criterion) == 2
        assert criterion[0].player_name == "Alice"

    def test_from_evaluations_wrapped_dict(self) -> None:
        wrapped = {"evaluations": _sample_eval_dict()}
        result = EvaluationResult.from_dict(wrapped)
        assert result.get_criteria_names() == ["natural_expression"]


class TestTeamAggregationService:
    def test_aggregation_summary_includes_metadata(self) -> None:
        svc = TeamAggregationService(_mappings())
        data = svc.build_aggregation_data([_sample_eval_dict()])
        assert data["summary"]["total_games_processed"] == 1
        assert set(data["summary"]["teams_found"]) == {"team_a", "team_b"}
        # description で並ぶ
        assert "発話表現は自然か" in data["summary"]["criteria_evaluated"]

    def test_aggregation_team_averages_by_description_key(self) -> None:
        svc = TeamAggregationService(_mappings())
        data = svc.build_aggregation_data([_sample_eval_dict()])
        # criteria 名は description に変換されている
        assert "発話表現は自然か" in data["team_averages"]["team_a"]
        assert data["team_averages"]["team_a"]["発話表現は自然か"] == 1.0
        assert data["team_averages"]["team_b"]["発話表現は自然か"] == 2.0

    def test_criteria_sorted_by_order(self) -> None:
        """複数 criteria を持つゲームを与えると order 順に並ぶ."""
        eval_dict = {
            "team_play": {
                "rankings": [
                    {"player_name": "A", "team": "x", "ranking": 1, "reasoning": "r"}
                ]
            },
            "natural_expression": {
                "rankings": [
                    {"player_name": "A", "team": "x", "ranking": 1, "reasoning": "r"}
                ]
            },
        }
        svc = TeamAggregationService(_mappings())
        data = svc.build_aggregation_data([eval_dict])
        criteria_order = data["summary"]["criteria_evaluated"]
        # order=1 が先、order=6 が後ろ
        assert criteria_order == ["発話表現は自然か", "チームプレイができているか"]

    def test_empty_input_yields_empty_summary(self) -> None:
        svc = TeamAggregationService(_mappings())
        data = svc.build_aggregation_data([])
        assert data["summary"]["total_games_processed"] == 0
        assert data["team_averages"] == {}
