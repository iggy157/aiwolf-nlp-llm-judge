"""Settings.yaml 専用ローダー."""

from pathlib import Path

from src.game.models import GameFormat
from src.utils.yaml_loader import YAMLLoader


class SettingsLoader:
    """settings.yamlファイルの読み込み専用クラス."""

    @staticmethod
    def load_game_format(settings_path: Path) -> GameFormat:
        """settings.yamlからゲーム形式設定を読み込む

        Args:
            settings_path: settings.yamlファイルのパス

        Returns:
            読み込まれたゲーム形式

        Raises:
            FileNotFoundError: 設定ファイルが見つからない場合
            ValueError: 設定ファイルの形式が不正な場合
        """
        settings_data = YAMLLoader.load_yaml(settings_path)

        # ゲーム形式設定を取得
        game_format_str = settings_data.get("game", {}).get("format", "main_match")

        try:
            return GameFormat(game_format_str)
        except ValueError:
            raise ValueError(f"Unknown game format: {game_format_str}")

    @staticmethod
    def get_evaluation_criteria_path(settings_path: Path) -> Path:
        """settings.yamlから評価基準ファイルのパスを取得

        Args:
            settings_path: settings.yamlファイルのパス

        Returns:
            評価基準ファイルの絶対パス

        Raises:
            FileNotFoundError: 設定ファイルが見つからない場合
            ValueError: 設定ファイルの形式が不正な場合
        """
        settings_data = YAMLLoader.load_yaml(settings_path)

        # evaluation_criteria のパスを取得
        evaluation_criteria_path = settings_data.get("path", {}).get(
            "evaluation_criteria"
        )
        if not evaluation_criteria_path:
            raise ValueError("evaluation_criteria path not found in settings")

        # 相対パスの場合、プロジェクトルートからの相対パスとして解釈
        if Path(evaluation_criteria_path).is_absolute():
            return Path(evaluation_criteria_path)
        else:
            # プロジェクトルートを取得（settings.yamlの親の親ディレクトリ）
            project_root = settings_path.parent.parent
            return project_root / evaluation_criteria_path
