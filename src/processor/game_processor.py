"""単一ゲームの処理を担当するクラス."""

import logging
from pathlib import Path
from typing import Any

from src.aiwolf_log.game_log import AIWolfGameLog
from src.llm.client import ModelConfig

from .models.config import ProcessingConfig
from .pipeline import (
    DataPreparationService,
    EvaluationExecutionService,
    ResultWritingService,
)
from .pipeline.game_context import GameContextBuilder

logger = logging.getLogger(__name__)


class GameProcessor:
    """単一ゲームの処理を担当するクラス.

    責任:
    - GameContext 構築（前処理）
    - 評価実行サービスの起動
    - 結果保存
    """

    SUCCESS_INDICATOR = "✓"
    FAILURE_INDICATOR = "✗"

    def __init__(self, config: dict[str, Any], model_config: ModelConfig) -> None:
        """GameProcessorを初期化

        Args:
            config: アプリケーション設定辞書（path.env / llm.prompt_yml / processing.encoding 等の plumbing）
            model_config: 使用するLLMモデルの設定

        Raises:
            ConfigurationError: 設定が不正な場合
        """
        self.config = config
        self.model_config = model_config

        self.processing_config = ProcessingConfig.from_config_dict(config)
        self.data_prep_service = DataPreparationService(config)
        self.result_service = ResultWritingService()

        self.evaluation_service = EvaluationExecutionService(
            config, model_config, self.processing_config
        )

        self.context_builder = GameContextBuilder(
            settings_path=self.data_prep_service.settings_path,
            evaluation_config=self.data_prep_service.load_evaluation_config(),
            reader_config=config,
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

            ctx = self.context_builder.build(game_log)

            evaluation_result = self.evaluation_service.execute_evaluations(
                ctx.game_info,
                ctx.criteria,
                ctx.formatted_data,
                ctx.character_info,
                ctx.agent_to_team_mapping,
            )

            if persist:
                if output_dir is None:
                    raise ValueError("output_dir is required when persist=True")
                self.result_service.save_results(
                    game_log, ctx.game_info, evaluation_result, output_dir
                )

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
