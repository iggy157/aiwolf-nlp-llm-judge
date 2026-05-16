"""データ準備・読み込みサービス."""

import logging
from pathlib import Path
from typing import Any

from src.aiwolf_log.game_log import AIWolfGameLog
from src.evaluation.loaders.criteria_loader import CriteriaLoader
from src.evaluation.loaders.settings_loader import SettingsLoader
from src.evaluation.models.config import EvaluationConfig
from src.game.detector import GameDetector
from src.game.models import GameInfo
from src.processor.models.exceptions import ConfigurationError, GameLogProcessingError

logger = logging.getLogger(__name__)


class DataPreparationService:
    """データ準備・読み込みを担当するサービス

    責任:
    - 評価設定の読み込み
    - ゲーム情報の検出
    - 設定からスレッド数の取得
    """

    DEFAULT_EVALUATION_WORKERS = 8

    def __init__(self, config: dict[str, Any]) -> None:
        """初期化

        Args:
            config: アプリケーション設定辞書

        Raises:
            ConfigurationError: 設定が不正な場合
        """
        self.config = config

        if "settings_path" not in config:
            raise ConfigurationError("settings_path is required in configuration")

        self.settings_path = Path(config["settings_path"])

    def load_evaluation_config(self) -> EvaluationConfig:
        """評価設定を読み込み

        Returns:
            評価設定オブジェクト

        Raises:
            ConfigurationError: 設定読み込みに失敗した場合
        """
        try:
            criteria_path = SettingsLoader.get_evaluation_criteria_path(
                self.settings_path
            )
            config = CriteriaLoader.load_evaluation_config(criteria_path)
            logger.debug(f"Loaded {len(config)} evaluation criteria")
            return config
        except Exception as e:
            raise ConfigurationError(f"Failed to load evaluation config: {e}") from e

    def detect_game_info(self, game_log: AIWolfGameLog) -> GameInfo:
        """ゲーム情報を検出

        Args:
            game_log: ゲームログ

        Returns:
            検出されたゲーム情報

        Raises:
            GameLogProcessingError: ゲーム情報の検出に失敗した場合
        """
        try:
            game_info = GameDetector.detect_game_format(
                game_log.log_path, self.settings_path
            )
            logger.debug(
                f"Detected game - format: {game_info.game_format.value}, "
                f"players: {game_info.player_count}, "
                f"werewolves: {game_info.werewolf_count}"
            )
            return game_info
        except Exception as e:
            raise GameLogProcessingError(f"Failed to detect game info: {e}") from e

    def get_evaluation_workers(self) -> int:
        """設定から評価用スレッド数を読み込み

        Returns:
            評価用スレッド数（デフォルト: DEFAULT_EVALUATION_WORKERS）

        Raises:
            ConfigurationError: 設定読み込みに失敗した場合
        """
        try:
            from src.utils.yaml_loader import YAMLLoader

            config_data = YAMLLoader.load_yaml(self.settings_path)
            evaluation_workers = config_data.get("processing", {}).get(
                "evaluation_workers", self.DEFAULT_EVALUATION_WORKERS
            )

            # 値の妥当性チェック
            if not isinstance(evaluation_workers, int) or evaluation_workers < 1:
                logger.warning(
                    f"Invalid evaluation_workers value: {evaluation_workers}, "
                    f"using default: {self.DEFAULT_EVALUATION_WORKERS}"
                )
                return self.DEFAULT_EVALUATION_WORKERS

            logger.debug(f"Loaded evaluation_workers: {evaluation_workers}")
            return evaluation_workers

        except Exception as e:
            logger.warning(
                f"Failed to load evaluation_workers: {e}, "
                f"using default: {self.DEFAULT_EVALUATION_WORKERS}"
            )
            return self.DEFAULT_EVALUATION_WORKERS
