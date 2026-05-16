from abc import ABC, abstractmethod
from pathlib import Path

from src.game.models import GameInfo
from src.evaluation.models import (
    EvaluationResult,
    EvaluationConfig,
    EvaluationLLMResponse,
)


class BaseEvaluator(ABC):
    """評価器の基底クラス."""

    def __init__(self, config: EvaluationConfig) -> None:
        """評価器を初期化.

        Args:
            config: 評価設定
        """
        self.config = config

    @abstractmethod
    def evaluate(self, csv_path: Path, game_info: GameInfo) -> EvaluationResult:
        """CSVファイルを評価.

        Args:
            csv_path: 評価対象のCSVファイルパス
            game_info: ゲーム情報

        Returns:
            EvaluationResult: 評価結果
        """
        pass

    def _create_evaluation_result(
        self,
        responses: dict[str, EvaluationLLMResponse],
    ) -> EvaluationResult:
        """評価結果オブジェクトを作成.

        Args:
            responses: 評価レスポンス辞書（基準名 -> レスポンス）

        Returns:
            EvaluationResult: 作成された評価結果
        """
        # EvaluationResultを作成し、レスポンスを追加
        result = EvaluationResult()
        for criteria_name, response in responses.items():
            result.add_response(criteria_name, response)

        return result

    def _validate_responses(
        self,
        responses: dict[str, EvaluationLLMResponse],
        game_info: GameInfo,
    ) -> None:
        """評価レスポンスの妥当性をチェック.

        Args:
            responses: 評価レスポンス辞書
            game_info: ゲーム情報

        Raises:
            ValueError: 評価レスポンスが不正な場合
        """
        expected_criteria = self.config.get_criteria_for_game(game_info)
        expected_names = {c.name for c in expected_criteria}

        # 不足している基準をチェック
        actual_names = set(responses.keys())
        missing_criteria = expected_names - actual_names
        if missing_criteria:
            raise ValueError(f"Missing evaluations for criteria: {missing_criteria}")

        # 余分な基準をチェック
        extra_criteria = actual_names - expected_names
        if extra_criteria:
            raise ValueError(f"Unexpected criteria in evaluations: {extra_criteria}")

        # 各評価の基本的な整合性チェック（rankingの重複など）
        for criteria_name, response in responses.items():
            rankings = [elem.ranking for elem in response.rankings]

            # ランキングに重複がないかチェック
            if len(set(rankings)) != len(rankings):
                raise ValueError(
                    f"Duplicate rankings found in criteria '{criteria_name}'"
                )
