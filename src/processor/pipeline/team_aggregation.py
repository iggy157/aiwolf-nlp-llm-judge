"""チーム集計を担当するサービス.

`BatchProcessor` 側に分散していたチーム集計関連ロジック
（criteria名→description変換、orderソート、dict→EvaluationResult復元）を集約する。
"""

import json
import logging
from pathlib import Path
from typing import Any

from src.evaluation.models.config import EvaluationConfig
from src.evaluation.models.result import EvaluationResult, TeamAggregator
from src.processor.pipeline.aggregation_output import AggregationOutputService

logger = logging.getLogger(__name__)


CriteriaMappings = dict[str, dict[str, Any]]


def build_criteria_mappings(
    evaluation_config: EvaluationConfig,
) -> CriteriaMappings:
    """評価設定から criteria_name → description / order のマッピングを構築."""
    return {
        "criteria_name_to_description": {
            criteria.name: criteria.description for criteria in evaluation_config
        },
        "criteria_name_to_order": {
            criteria.name: criteria.order for criteria in evaluation_config
        },
    }


class TeamAggregationService:
    """チーム集計を行うサービス.

    `__init__` で criteria_mappings を1度だけ受け取り、それを使い回す。
    """

    def __init__(
        self,
        criteria_mappings: CriteriaMappings,
        output_service: AggregationOutputService | None = None,
    ) -> None:
        self.criteria_mappings = criteria_mappings
        self.output_service = output_service or AggregationOutputService()

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def generate_and_save(
        self,
        evaluation_dicts: list[dict],
        output_dir: Path,
        model_id: str | None = None,
    ) -> None:
        """評価結果辞書のリストから集計を生成し、JSON/CSV で保存."""
        label = f"[{model_id}] " if model_id else ""
        logger.info(f"{label}Generating team aggregation")
        try:
            aggregation_data = self.build_aggregation_data(evaluation_dicts)
            self.output_service.save_both(aggregation_data, output_dir)
        except Exception as e:
            logger.error(
                f"{label}Failed to generate team aggregation: {e}", exc_info=True
            )

    def regenerate_for_model_dir(self, model_dir: Path) -> None:
        """既存の `*_result.json` から集計のみを再生成."""
        result_files = list(model_dir.glob("*_result.json"))
        if not result_files:
            logger.warning(f"No *_result.json found in {model_dir}; skipping")
            return
        logger.info(
            f"[{model_dir.name}] Found {len(result_files)} evaluation result files"
        )

        evaluation_dicts: list[dict] = []
        for result_file in result_files:
            try:
                with result_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if "evaluations" in data:
                    evaluation_dicts.append(data["evaluations"])
                else:
                    logger.warning(
                        f"No 'evaluations' field in {result_file.name}; skipping"
                    )
            except Exception as e:
                logger.error(f"Failed to load {result_file.name}: {e}")

        if not evaluation_dicts:
            logger.error(f"[{model_dir.name}] No valid evaluation results loaded")
            return

        try:
            aggregation_data = self.build_aggregation_data(evaluation_dicts)
            self.output_service.save_both(aggregation_data, model_dir)
            logger.info(f"[{model_dir.name}] Team aggregation regenerated")
        except Exception as e:
            logger.error(
                f"[{model_dir.name}] Failed to regenerate aggregation: {e}",
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Building aggregation data
    # ------------------------------------------------------------------

    def build_aggregation_data(self, evaluation_dicts: list[dict]) -> dict[str, Any]:
        """評価辞書のリストからチーム集計データを構築."""
        aggregator = TeamAggregator()
        for evaluation_dict in evaluation_dicts:
            aggregator.add_game_result(EvaluationResult.from_dict(evaluation_dict))

        team_averages = aggregator.calculate_team_averages()
        team_counts = aggregator.get_team_count_by_criteria()

        team_averages_with_descriptions = self._convert_names_to_descriptions(
            team_averages
        )
        team_counts_with_descriptions = self._convert_names_to_descriptions(team_counts)
        criteria_evaluated = self._build_sorted_criteria_list(
            team_averages_with_descriptions
        )

        return {
            "team_averages": team_averages_with_descriptions,
            "team_sample_counts": team_counts_with_descriptions,
            "summary": {
                "total_games_processed": len(evaluation_dicts),
                "teams_found": list(team_averages_with_descriptions.keys()),
                "criteria_evaluated": criteria_evaluated,
            },
        }

    # ------------------------------------------------------------------
    # Helpers (privately scoped)
    # ------------------------------------------------------------------

    def _convert_names_to_descriptions(
        self, data: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        name_to_description = self.criteria_mappings["criteria_name_to_description"]
        name_to_order = self.criteria_mappings["criteria_name_to_order"]
        converted: dict[str, dict[str, Any]] = {}
        for team, criteria_dict in data.items():
            sorted_items = sorted(
                criteria_dict.items(),
                key=lambda kv: name_to_order.get(kv[0], 999),
            )
            converted[team] = {
                name_to_description.get(name, name): value
                for name, value in sorted_items
            }
        return converted

    def _build_sorted_criteria_list(
        self, team_averages_with_descriptions: dict[str, dict[str, Any]]
    ) -> list[str]:
        if not team_averages_with_descriptions:
            return []
        name_to_description = self.criteria_mappings["criteria_name_to_description"]
        name_to_order = self.criteria_mappings["criteria_name_to_order"]
        description_to_name = {v: k for k, v in name_to_description.items()}

        first_team = next(iter(team_averages_with_descriptions.values()), {})
        order_and_desc: list[tuple[int, str]] = []
        for description in first_team.keys():
            name = description_to_name.get(description, description)
            order_and_desc.append((name_to_order.get(name, 999), description))
        return [desc for _, desc in sorted(order_and_desc)]
