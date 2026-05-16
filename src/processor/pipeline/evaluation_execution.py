"""評価実行・並列処理サービス."""

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from src.evaluation.models.config import EvaluationConfig
from src.evaluation.models.criteria import EvaluationCriteria
from src.evaluation.models.llm_response import EvaluationLLMResponse
from src.evaluation.models.result import EvaluationResult, CriteriaEvaluationResult
from src.game.models import GameInfo
from src.llm.client import ModelConfig
from src.llm.evaluator import Evaluator
from src.processor.models.exceptions import EvaluationExecutionError

logger = logging.getLogger(__name__)


class EvaluationExecutionService:
    """評価実行・並列処理を担当するサービス

    責任:
    - マルチスレッド評価の管理
    - 単一評価基準の実行
    """

    def __init__(
        self,
        config: dict[str, Any],
        model_config: ModelConfig,
        max_evaluation_threads: int = 8,
    ) -> None:
        """初期化

        Args:
            config: アプリケーション設定辞書
            model_config: 使用するLLMモデルの設定
            max_evaluation_threads: 評価用最大スレッド数
        """
        self.config = config
        self.model_config = model_config
        self.max_evaluation_threads = max_evaluation_threads
        # 設定からmax_retriesを読み取り（デフォルトは3）
        self.max_retries = config.get("processing", {}).get("max_retries", 3)
        # キャッシュは provider 側で各々挙動するが、ここでスイッチ可能にしておく
        self.enable_caching = bool(
            config.get("processing", {}).get("enable_caching", True)
        )

    @staticmethod
    def _extract_player_names_from_character_info(character_info: str) -> set[str]:
        """キャラクター情報文字列からプレイヤー名を抽出

        Args:
            character_info: "- agent_name: profile" 形式の文字列

        Returns:
            プレイヤー名のセット
        """
        if not character_info:
            return set()

        player_names = set()
        # "- name: profile" の形式から name を抽出
        for line in character_info.split("\n"):
            match = re.match(r"^-\s+([^:]+):", line.strip())
            if match:
                player_names.add(match.group(1).strip())

        return player_names

    def execute_evaluations(
        self,
        evaluation_config: EvaluationConfig,
        game_info: GameInfo,
        formatted_data: list[dict[str, Any]],
        character_info: str,
        agent_to_team_mapping: dict[str, str],
    ) -> EvaluationResult:
        """評価を並列実行

        Args:
            evaluation_config: 評価設定
            game_info: ゲーム情報
            formatted_data: フォーマット済みログデータ
            character_info: キャラクター情報
            agent_to_team_mapping: エージェント名→チーム名のマッピング

        Returns:
            評価結果

        Raises:
            EvaluationExecutionError: 評価実行に失敗した場合
        """
        criteria_for_game = evaluation_config.get_criteria_for_game(game_info)

        if not criteria_for_game:
            logger.warning(
                f"No evaluation criteria found for game "
                f"(players={game_info.player_count}, "
                f"werewolves={game_info.werewolf_count})"
            )
            return EvaluationResult()

        logger.info(f"Starting evaluation for {len(criteria_for_game)} criteria")

        evaluator = Evaluator(self.config, self.model_config)
        cache_handle = None
        if self.enable_caching:
            cache_handle = evaluator.open_cache(character_info)
            if cache_handle is not None:
                logger.debug(
                    f"[{self.model_config.id}] opened cache: "
                    f"{cache_handle.resource_name}"
                )

        try:
            evaluation_result = EvaluationResult()

            max_workers = min(len(criteria_for_game), self.max_evaluation_threads)

            valid_player_names = self._extract_player_names_from_character_info(
                character_info
            )

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_criteria = {
                    executor.submit(
                        self._evaluate_criterion,
                        criteria,
                        formatted_data,
                        evaluator,
                        character_info,
                        game_info.player_count,
                        valid_player_names,
                        self.max_retries,
                        cache_handle,
                    ): criteria
                    for criteria in criteria_for_game
                }

                for future in as_completed(future_to_criteria):
                    criteria = future_to_criteria[future]
                    try:
                        criteria_name, llm_response = future.result()
                        criteria_result = CriteriaEvaluationResult.from_llm_response(
                            criteria_name, llm_response, agent_to_team_mapping
                        )
                        evaluation_result.append(criteria_result)
                        logger.debug(f"Completed evaluation: {criteria_name}")
                    except Exception as e:
                        error_msg = f"Evaluation failed for {criteria.name}: {e}"
                        logger.error(error_msg, exc_info=True)
                        raise EvaluationExecutionError(error_msg) from e

            logger.info(f"Completed all {len(criteria_for_game)} evaluations")
            return evaluation_result

        except Exception as e:
            if isinstance(e, EvaluationExecutionError):
                raise
            raise EvaluationExecutionError(f"Failed to execute evaluations: {e}") from e
        finally:
            evaluator.close_cache(cache_handle)

    @staticmethod
    def _evaluate_criterion(
        criteria: EvaluationCriteria,
        formatted_data: list[dict[str, Any]],
        evaluator: Evaluator,
        character_info: str,
        player_count: int,
        valid_player_names: set[str],
        max_retries: int,
        cache_handle=None,
    ) -> tuple[str, EvaluationLLMResponse]:
        """単一評価基準の評価を実行（バリデーション付きで再試行）

        Args:
            criteria: 評価基準
            formatted_data: フォーマット済みログデータ
            evaluator: LLM評価器
            character_info: キャラクター情報
            player_count: 期待するプレイヤー数
            valid_player_names: 有効なプレイヤー名のセット
            max_retries: 最大再試行回数

        Returns:
            (評価基準名, LLMレスポンス)のタプル

        Raises:
            ValueError: 最大再試行回数後もバリデーションに失敗した場合
        """
        logger.debug(f"Evaluating: {criteria.name}")

        for attempt in range(max_retries + 1):
            try:
                # LLMから基本的な応答を取得
                llm_response = evaluator.evaluation(
                    criteria=criteria,
                    log=formatted_data,
                    output_structure=EvaluationLLMResponse,
                    character_info=character_info,
                    cache_handle=cache_handle,
                )

                # バリデーション付きで再作成
                validated_response = EvaluationLLMResponse.create_with_validation(
                    rankings=llm_response.rankings,
                    player_count=player_count,
                    valid_player_names=valid_player_names,
                )

                logger.debug(f"Successfully validated evaluation: {criteria.name}")
                return criteria.name, validated_response

            except ValueError as e:
                attempt_msg = f"attempt {attempt + 1}/{max_retries + 1}"
                logger.warning(
                    f"Validation failed for {criteria.name} ({attempt_msg}): {e}"
                )

                if attempt == max_retries:
                    error_msg = f"Failed to get valid evaluation for {criteria.name} after {max_retries + 1} attempts"
                    logger.error(error_msg)
                    raise ValueError(error_msg) from e

                logger.info(f"Retrying evaluation for {criteria.name}...")

        # このコードに到達することはないが、型チェッカーのために必要
        raise RuntimeError("Unexpected code path reached")
