"""Evaluation criteria.yaml 専用ローダー."""

from pathlib import Path
from typing import Any

from src.evaluation.models import (
    ApplicableWhen,
    EvaluationConfig,
    EvaluationCriteria,
    RankingType,
)
from src.utils.yaml_loader import YAMLLoader


class CriteriaLoader:
    """evaluation_criteria.yamlファイルの読み込み専用クラス."""

    @staticmethod
    def load_evaluation_config(config_path: Path) -> EvaluationConfig:
        """評価設定ファイルを読み込んでEvaluationConfigオブジェクトを作成

        Args:
            config_path: 設定ファイルのパス

        Returns:
            読み込まれた評価設定

        Raises:
            FileNotFoundError: 設定ファイルが見つからない場合
            ValueError: 設定ファイルの形式が不正な場合
        """
        config_data = YAMLLoader.load_yaml(config_path)

        criteria_data = config_data.get("criteria")
        if criteria_data is None:
            raise ValueError(
                f"'criteria' key not found in {config_path}. "
                "evaluation_criteria.yaml must define a top-level 'criteria' list."
            )
        if not isinstance(criteria_data, list):
            raise ValueError(
                f"'criteria' must be a list, got {type(criteria_data).__name__}"
            )

        criteria = [
            CriteriaLoader._load_criteria_dict(criteria_dict)
            for criteria_dict in criteria_data
        ]

        return EvaluationConfig(criteria)

    @staticmethod
    def _load_criteria_dict(criteria_dict: dict[str, Any]) -> EvaluationCriteria:
        """評価基準辞書を読み込んでEvaluationCriteriaオブジェクトを作成

        Args:
            criteria_dict: YAML から読み込まれた評価基準データ

        Returns:
            評価基準オブジェクト

        Raises:
            ValueError: 設定データが不正な場合
        """
        try:
            name = criteria_dict["name"]
            description = criteria_dict["description"]
            ranking_type = criteria_dict["ranking_type"]
            order = criteria_dict.get("order", 0)

            if ranking_type == "ordinal":
                ranking_type_enum = RankingType.ORDINAL
            elif ranking_type == "comparative":
                ranking_type_enum = RankingType.COMPARATIVE
            else:
                raise ValueError(f"Invalid ranking type: {ranking_type}")

            applicable_when = CriteriaLoader._load_applicable_when(
                criteria_dict.get("applicable_when")
            )

            return EvaluationCriteria(
                name=name,
                description=description,
                ranking_type=ranking_type_enum,
                order=order,
                applicable_when=applicable_when,
            )

        except KeyError as e:
            raise ValueError(f"Missing required field in criteria: {e}")
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid criteria data: {e}")

    @staticmethod
    def _load_applicable_when(
        applicable_when_data: dict[str, Any] | None,
    ) -> ApplicableWhen | None:
        """applicable_whenセクションを読み込み

        Args:
            applicable_when_data: applicable_whenディクショナリ、もしくはNone

        Returns:
            ApplicableWhenオブジェクト、もしくは指定がなければNone

        Raises:
            ValueError: 未知のキーや不正な値が含まれる場合
        """
        if applicable_when_data is None:
            return None
        if not isinstance(applicable_when_data, dict):
            raise ValueError(
                f"'applicable_when' must be a mapping, got "
                f"{type(applicable_when_data).__name__}"
            )

        known_keys = {"werewolf_count_gte"}
        unknown = set(applicable_when_data.keys()) - known_keys
        if unknown:
            raise ValueError(
                f"Unknown applicable_when keys: {sorted(unknown)}. "
                f"Supported keys: {sorted(known_keys)}"
            )

        werewolf_count_gte = applicable_when_data.get("werewolf_count_gte")
        if werewolf_count_gte is not None and (
            not isinstance(werewolf_count_gte, int) or werewolf_count_gte < 0
        ):
            raise ValueError(
                f"'werewolf_count_gte' must be a non-negative integer, got "
                f"{werewolf_count_gte!r}"
            )

        return ApplicableWhen(werewolf_count_gte=werewolf_count_gte)
