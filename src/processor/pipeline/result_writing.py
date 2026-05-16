"""結果保存・チームマッピングサービス."""

import json
import logging
from pathlib import Path
from typing import Any

from src.aiwolf_log.game_log import AIWolfGameLog
from src.evaluation.models.result import EvaluationResult
from src.game.models import GameInfo

logger = logging.getLogger(__name__)


class ResultWritingService:
    """結果保存・チームマッピングを担当するサービス

    責任:
    - 評価結果データの構築
    - チームマッピングの処理
    - ファイル保存
    """

    def save_results(
        self,
        game_log: AIWolfGameLog,
        game_info: GameInfo,
        evaluation_result: EvaluationResult,
        output_dir: Path,
    ) -> None:
        """評価結果を保存

        Args:
            game_log: ゲームログ
            game_info: ゲーム情報
            evaluation_result: 評価結果
            output_dir: 出力ディレクトリ
        """
        result_data = self._build_result_data(game_log, game_info, evaluation_result)
        output_path = output_dir / f"{game_log.game_id}_result.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)

        logger.debug(f"Saved results to: {output_path}")

    def _build_result_data(
        self,
        game_log: AIWolfGameLog,
        game_info: GameInfo,
        evaluation_result: EvaluationResult,
    ) -> dict[str, Any]:
        """評価結果データを構築

        Args:
            game_log: ゲームログ
            game_info: ゲーム情報
            evaluation_result: 評価結果

        Returns:
            構築された結果データ辞書
        """
        # デバッグ用ログ
        self._log_debug_info(evaluation_result)

        result_data = {
            "game_id": game_log.game_id,
            "game_info": {
                "format": game_info.game_format.value,
                "player_count": game_info.player_count,
                "werewolf_count": game_info.werewolf_count,
            },
            "evaluations": {},
        }

        for criteria_result in evaluation_result:
            result_data["evaluations"][criteria_result.criteria_name] = {
                "rankings": [
                    {
                        "player_name": elem.player_name,
                        "team": elem.team,  # 既にチーム情報が含まれている
                        "ranking": elem.ranking,
                        "reasoning": elem.reasoning,
                    }
                    for elem in criteria_result
                ]
            }

        return result_data

    def _log_debug_info(self, evaluation_result: EvaluationResult) -> None:
        """デバッグ情報をログ出力

        Args:
            evaluation_result: 評価結果
        """

        if evaluation_result:
            sample_criteria_result = evaluation_result[0]
            if sample_criteria_result:
                sample_players = [
                    elem.player_name for elem in sample_criteria_result[:3]
                ]
                logger.debug(f"Sample player names from LLM: {sample_players}")
