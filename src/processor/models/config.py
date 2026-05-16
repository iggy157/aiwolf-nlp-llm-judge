"""プロセッサー処理設定を表すデータクラス."""

import multiprocessing as mp
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.game.models import GameFormat
from src.llm.client import ModelConfig
from src.llm.factory import model_config_from_dict

from .exceptions import ConfigurationError


@dataclass(frozen=True)
class ProcessingConfig:
    """処理設定を表すデータクラス.

    Attributes:
        input_dir: 入力ディレクトリのパス
        output_root: 出力ディレクトリのルート（タイムスタンプ配下が実行毎に作られる）
        max_workers: 並列処理の最大ワーカー数
        game_format: ゲーム形式
        models: 実行対象モデル一覧
        parallel_models: True で複数モデルを同時並行実行
        dry_run: True で本実行前に試運転（1ゲーム × 全評価基準）
        dry_run_strict: True で試運転失敗時に全体中断
    """

    input_dir: Path
    output_root: Path
    max_workers: int
    game_format: GameFormat
    models: list[ModelConfig] = field(default_factory=list)
    parallel_models: bool = False
    dry_run: bool = True
    dry_run_strict: bool = False
    use_batch_api: bool = False
    batch_poll_interval_seconds: float = 60.0
    batch_max_wait_seconds: float = 86400.0

    @staticmethod
    def from_config_dict(config: dict[str, Any]) -> "ProcessingConfig":
        """設定辞書から処理設定を作成.

        Args:
            config: アプリケーション設定辞書

        Returns:
            処理設定オブジェクト

        Raises:
            ConfigurationError: 設定が不正な場合
        """
        try:
            processing_config = config["processing"]
            input_dir = Path(processing_config["input_dir"])
            output_root = Path(processing_config["output_dir"])
            max_workers = processing_config.get("max_workers") or mp.cpu_count()
            parallel_models = bool(processing_config.get("parallel_models", False))
            dry_run = bool(processing_config.get("dry_run", True))
            dry_run_strict = bool(processing_config.get("dry_run_strict", False))
            use_batch_api = bool(processing_config.get("use_batch_api", False))
            batch_poll_interval_seconds = float(
                processing_config.get("batch_poll_interval_seconds", 60.0)
            )
            batch_max_wait_seconds = float(
                processing_config.get("batch_max_wait_seconds", 86400.0)
            )

            game_format_str = config.get("game", {}).get("format", "self_match")
            try:
                game_format = GameFormat(game_format_str)
            except ValueError as e:
                valid_formats = [fmt.value for fmt in GameFormat]
                raise ConfigurationError(
                    f"Invalid game format: '{game_format_str}'. "
                    f"Valid values are: {valid_formats}"
                ) from e

            models = ProcessingConfig._load_models(config)

            return ProcessingConfig(
                input_dir=input_dir,
                output_root=output_root,
                max_workers=max_workers,
                game_format=game_format,
                models=models,
                parallel_models=parallel_models,
                dry_run=dry_run,
                dry_run_strict=dry_run_strict,
                use_batch_api=use_batch_api,
                batch_poll_interval_seconds=batch_poll_interval_seconds,
                batch_max_wait_seconds=batch_max_wait_seconds,
            )

        except KeyError as e:
            raise ConfigurationError(f"Missing required config key: {e}") from e

    @staticmethod
    def _load_models(config: dict[str, Any]) -> list[ModelConfig]:
        """llm.models 配列を ModelConfig のリストに変換."""
        llm_config = config.get("llm", {})
        models_data = llm_config.get("models")
        if not models_data:
            raise ConfigurationError(
                "llm.models is required (at least one model must be configured)"
            )
        if not isinstance(models_data, list):
            raise ConfigurationError(
                f"llm.models must be a list, got {type(models_data).__name__}"
            )

        models: list[ModelConfig] = []
        seen_ids: set[str] = set()
        for entry in models_data:
            if not isinstance(entry, dict):
                raise ConfigurationError(
                    f"Each entry in llm.models must be a mapping, got "
                    f"{type(entry).__name__}"
                )
            try:
                model_config = model_config_from_dict(entry)
            except ValueError as e:
                raise ConfigurationError(str(e)) from e

            if model_config.id in seen_ids:
                raise ConfigurationError(
                    f"Duplicate model id in llm.models: {model_config.id}"
                )
            seen_ids.add(model_config.id)
            models.append(model_config)
        return models

    def filter_models(self, model_ids: list[str] | None) -> "ProcessingConfig":
        """指定された id のモデルだけに絞り込んだ新しい ProcessingConfig を返す."""
        if not model_ids:
            return self
        wanted = set(model_ids)
        available = {m.id for m in self.models}
        unknown = wanted - available
        if unknown:
            raise ConfigurationError(
                f"Unknown model ids: {sorted(unknown)}. "
                f"Available: {sorted(available)}"
            )
        filtered = [m for m in self.models if m.id in wanted]
        return ProcessingConfig(
            input_dir=self.input_dir,
            output_root=self.output_root,
            max_workers=self.max_workers,
            game_format=self.game_format,
            models=filtered,
            parallel_models=self.parallel_models,
            dry_run=self.dry_run,
            dry_run_strict=self.dry_run_strict,
            use_batch_api=self.use_batch_api,
            batch_poll_interval_seconds=self.batch_poll_interval_seconds,
            batch_max_wait_seconds=self.batch_max_wait_seconds,
        )
