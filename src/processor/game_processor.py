"""単一ゲームの処理を担当するクラス（リファクタリング後）."""

import logging
from pathlib import Path
from typing import Any

from src.aiwolf_log.game_log import AIWolfGameLog
from src.llm.client import ModelConfig

from .pipeline import (
    DataPreparationService,
    EvaluationExecutionService,
    LogFormattingService,
    ResultWritingService,
)

logger = logging.getLogger(__name__)


class GameProcessor:
    """単一ゲームの処理を担当するクラス（リファクタリング後）

    このクラスは、各サービスを組み合わせてゲームログの評価処理を
    オーケストレートする責任を持ちます。

    責任:
    - 処理フローの管理
    - サービス間の調整
    - エラーハンドリング
    """

    # クラス定数
    SUCCESS_INDICATOR = "✓"
    FAILURE_INDICATOR = "✗"

    def __init__(self, config: dict[str, Any], model_config: ModelConfig) -> None:
        """GameProcessorを初期化

        Args:
            config: アプリケーション設定辞書
            model_config: 使用するLLMモデルの設定

        Raises:
            ConfigurationError: 設定が不正な場合
        """
        self.config = config
        self.model_config = model_config

        # 各サービスを初期化
        self.data_prep_service = DataPreparationService(config)
        self.log_formatting_service = LogFormattingService(config)
        self.result_service = ResultWritingService()

        # 評価用スレッド数を取得して評価サービスを初期化
        max_threads = self.data_prep_service.get_evaluation_workers()
        self.evaluation_service = EvaluationExecutionService(
            config, model_config, max_threads
        )

    def process(
        self,
        game_log: AIWolfGameLog,
        output_dir: Path | None,
        persist: bool = True,
    ) -> tuple[bool, dict | None]:
        """ゲームログを処理して評価結果を出力

        Args:
            game_log: 処理対象のゲームログ
            output_dir: 結果出力ディレクトリ（persist=False なら無視可能）
            persist: True で結果JSONをファイル保存する。試運転（dry-run）では False。

        Returns:
            (処理が成功したかどうか, 評価結果辞書またはNone)
        """
        try:
            logger.info(
                f"[{self.model_config.id}] Processing game: {game_log.game_id}"
            )

            # 1. 設定とゲーム情報の準備
            evaluation_config = self.data_prep_service.load_evaluation_config()
            game_info = self.data_prep_service.detect_game_info(game_log)

            # 2. ログデータのフォーマット変換
            formatted_data = self.log_formatting_service.format_game_log(
                game_log, game_info
            )

            # 3. キャラクター情報の取得
            character_info = self.log_formatting_service.get_character_info(game_log)

            # 4. チームマッピングの取得
            agent_to_team_mapping = game_log.get_agent_to_team_mapping()

            # 5. 評価実行（マルチスレッド）
            evaluation_result = self.evaluation_service.execute_evaluations(
                evaluation_config,
                game_info,
                formatted_data,
                character_info,
                agent_to_team_mapping,
            )

            # 6. 結果保存（試運転時はスキップ）
            if persist:
                if output_dir is None:
                    raise ValueError("output_dir is required when persist=True")
                self.result_service.save_results(
                    game_log, game_info, evaluation_result, output_dir
                )

            # 7. 評価結果を辞書形式に変換
            evaluation_dict = evaluation_result.to_dict()

            logger.info(
                f"[{self.model_config.id}] Successfully processed game: "
                f"{game_log.game_id}"
            )
            return True, evaluation_dict

        except (FileNotFoundError, ValueError, KeyError) as e:
            logger.error(
                f"[{self.model_config.id}] Expected error processing game "
                f"{game_log.game_id}: {e}"
            )
            return False, None
        except Exception as e:
            logger.error(
                f"[{self.model_config.id}] Unexpected error processing game "
                f"{game_log.game_id}: {e}",
                exc_info=True,
            )
            return False, None
