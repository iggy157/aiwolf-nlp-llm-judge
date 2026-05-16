"""バッチ処理を管理するクラス（複数モデル対応）."""

import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from src.aiwolf_log import AIWolfGameLog
from src.llm.client import ModelConfig
from src.utils.game_log_finder import find_all_game_logs

from .game_processor import GameProcessor
from .models import ProcessingConfig, ProcessingResult
from .models.exceptions import ConfigurationError
from .pipeline.aggregation_output import AggregationOutputService
from .pipeline.team_aggregation import (
    TeamAggregationService,
    build_criteria_mappings,
)
from .run_directory import RunDirectory

logger = logging.getLogger(__name__)


# 試運転（dry-run）の結果サマリ
class _DryRunOutcome:
    __slots__ = ("model_id", "ok", "message")

    def __init__(self, model_id: str, ok: bool, message: str = "") -> None:
        self.model_id = model_id
        self.ok = ok
        self.message = message


class BatchProcessor:
    """複数モデル × 複数ゲームのバッチ処理を統括."""

    def __init__(
        self,
        config: dict[str, Any],
        model_ids_filter: list[str] | None = None,
    ) -> None:
        """BatchProcessorを初期化.

        Args:
            config: アプリケーション設定辞書
            model_ids_filter: CLI で --models 指定された場合のIDフィルタ
        """
        self.config = config
        self.processing_config = ProcessingConfig.from_config_dict(config).filter_models(
            model_ids_filter
        )
        if not self.processing_config.models:
            raise ConfigurationError("No models configured after filtering")

        self.aggregation_output = AggregationOutputService()
        # 集計用 criteria_mappings は実行毎に1回だけロードする（旧実装ではモデル毎に再パースしていた）
        self._aggregation_service = TeamAggregationService(
            criteria_mappings=self._load_criteria_mappings(),
            output_service=self.aggregation_output,
        )

    def _load_criteria_mappings(self) -> dict:
        """評価基準ファイルから criteria_mappings を構築（init で1回だけ呼ぶ）."""
        from src.processor.pipeline import DataPreparationService

        config_with_settings = self.config.copy()
        if "settings_path" not in config_with_settings:
            criteria_path = Path(
                self.config.get("path", {}).get(
                    "evaluation_criteria", "config/evaluation_criteria.yaml"
                )
            )
            settings_path = criteria_path.parent / "settings.yaml"
            config_with_settings["settings_path"] = str(settings_path)

        data_prep_service = DataPreparationService(config_with_settings)
        return build_criteria_mappings(data_prep_service.load_evaluation_config())

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def process_all_games(self, dry_run_only: bool = False) -> ProcessingResult:
        """全モデル × 全ゲームを処理.

        Args:
            dry_run_only: True で試運転だけ実施し本実行はしない

        Returns:
            全モデル合算の処理結果
        """
        logger.info(
            f"Starting batch run with {len(self.processing_config.models)} model(s): "
            f"{[m.id for m in self.processing_config.models]}"
        )

        game_logs = self._find_game_logs()
        if not game_logs:
            logger.warning("No game logs found")
            return ProcessingResult()

        run_dir = RunDirectory.create(self.processing_config.output_root)
        logger.info(f"Run directory: {run_dir}")

        # 試運転
        eligible_models = self.processing_config.models
        if self.processing_config.dry_run:
            eligible_models = self._dry_run(game_logs[0], eligible_models)
            if not eligible_models:
                logger.error("All models failed dry-run; aborting")
                self._write_run_metadata(run_dir, game_logs, eligible_models, [])
                return ProcessingResult()

        if dry_run_only:
            logger.info("dry-run-only mode; skipping full execution")
            self._write_run_metadata(run_dir, game_logs, eligible_models, [])
            return ProcessingResult(
                total=len(eligible_models),
                completed=len(eligible_models),
            )

        # 本実行
        per_model_results = self._run_all_models(eligible_models, game_logs, run_dir)
        self._write_run_metadata(
            run_dir, game_logs, eligible_models, list(per_model_results.keys())
        )

        # 合算
        total = sum(r.total for r in per_model_results.values())
        completed = sum(r.completed for r in per_model_results.values())
        failed = sum(r.failed for r in per_model_results.values())
        aggregated = ProcessingResult(total=total, completed=completed, failed=failed)
        for r in per_model_results.values():
            aggregated.evaluation_results.extend(r.evaluation_results)
        self._log_processing_summary(aggregated)
        return aggregated

    def regenerate_aggregation_only(self, run_dir: Path | None = None) -> None:
        """既存の評価結果JSONからチーム集計を再生成.

        Args:
            run_dir: 対象の <output_root>/<timestamp> ディレクトリ。
                     None の場合は output_root 直下で最新の timestamp ディレクトリを使う。
        """
        if run_dir is None:
            latest = RunDirectory.latest_under(self.processing_config.output_root)
            if latest is None:
                logger.error(
                    f"No timestamped run directory found under "
                    f"{self.processing_config.output_root}"
                )
                return
            run_dir_obj = latest
            logger.info(f"Using latest run directory: {run_dir_obj}")
        else:
            if not run_dir.is_dir():
                logger.error(f"Run directory does not exist: {run_dir}")
                return
            run_dir_obj = RunDirectory(run_dir)

        model_dirs = run_dir_obj.iter_model_dirs()
        if not model_dirs:
            logger.warning(f"No model subdirectories found in {run_dir_obj}")
            return

        for model_dir in model_dirs:
            self._regenerate_aggregation_for_model_dir(model_dir)

    # ------------------------------------------------------------------
    # Run-level helpers
    # ------------------------------------------------------------------

    def _find_game_logs(self) -> list[AIWolfGameLog]:
        logger.info(f"Searching for game logs in: {self.processing_config.input_dir}")
        game_logs = find_all_game_logs(self.processing_config.input_dir)
        logger.info(f"Found {len(game_logs)} game logs")
        return game_logs

    def _write_run_metadata(
        self,
        run_dir: RunDirectory,
        game_logs: list[AIWolfGameLog],
        eligible_models: list[ModelConfig],
        executed_model_ids: list[str],
    ) -> None:
        run_dir.write_metadata(
            input_dir=self.processing_config.input_dir,
            game_logs=game_logs,
            configured_models=self.processing_config.models,
            eligible_models=eligible_models,
            executed_model_ids=executed_model_ids,
            parallel_models=self.processing_config.parallel_models,
            dry_run=self.processing_config.dry_run,
            dry_run_strict=self.processing_config.dry_run_strict,
            use_batch_api=self.processing_config.use_batch_api,
        )

    def _log_processing_summary(self, result: ProcessingResult) -> None:
        logger.info(
            f"Batch processing completed - "
            f"Total: {result.total}, Success: {result.completed}, "
            f"Failed: {result.failed}, Success rate: {result.success_rate:.2%}"
        )

    # ------------------------------------------------------------------
    # Dry-run
    # ------------------------------------------------------------------

    def _dry_run(
        self,
        sample_game_log: AIWolfGameLog,
        models: list[ModelConfig],
    ) -> list[ModelConfig]:
        """各モデルに対して 1 ゲーム × 全評価基準で疎通確認.

        Args:
            sample_game_log: 試運転に使う1ゲーム（評価結果は破棄）
            models: 試運転対象のモデル一覧

        Returns:
            試運転を通過したモデルだけのリスト
        """
        logger.info(
            f"=== Dry-run with sample game '{sample_game_log.game_id}' "
            f"for {len(models)} model(s) ==="
        )

        outcomes: list[_DryRunOutcome] = []

        if self.processing_config.parallel_models and len(models) > 1:
            with ThreadPoolExecutor(max_workers=len(models)) as pool:
                future_to_model = {
                    pool.submit(self._dry_run_single, sample_game_log, m): m
                    for m in models
                }
                for future in as_completed(future_to_model):
                    outcomes.append(future.result())
        else:
            for m in models:
                outcomes.append(self._dry_run_single(sample_game_log, m))

        outcomes.sort(key=lambda o: o.model_id)
        passed_ids: list[str] = []
        failed_ids: list[tuple[str, str]] = []
        for o in outcomes:
            if o.ok:
                logger.info(f"  ✓ {o.model_id}: dry-run OK")
                passed_ids.append(o.model_id)
            else:
                logger.error(f"  ✗ {o.model_id}: dry-run FAILED: {o.message}")
                failed_ids.append((o.model_id, o.message))

        if failed_ids and self.processing_config.dry_run_strict:
            raise RuntimeError(
                f"Dry-run failed for models (strict mode): "
                f"{[mid for mid, _ in failed_ids]}"
            )

        eligible = [m for m in models if m.id in set(passed_ids)]
        logger.info(
            f"=== Dry-run complete: {len(eligible)}/{len(models)} model(s) eligible "
            f"for full run ==="
        )
        return eligible

    def _dry_run_single(
        self, game_log: AIWolfGameLog, model_config: ModelConfig
    ) -> _DryRunOutcome:
        # 試運転は必ず同期パスで実施（バッチAPI課金/24h待ちを避ける）。
        # バッチモードでも疎通確認は同期で十分。
        try:
            processor = GameProcessor(self.config, model_config)
            success, _ = processor.process(game_log, output_dir=None, persist=False)
            if success:
                return _DryRunOutcome(model_config.id, ok=True)
            return _DryRunOutcome(
                model_config.id,
                ok=False,
                message="process() returned failure",
            )
        except ConfigurationError:
            # 設定ミスは握りつぶさず即時伝播（dry_run_strict と無関係に致命的）
            raise
        except Exception as e:
            return _DryRunOutcome(model_config.id, ok=False, message=str(e))

    # ------------------------------------------------------------------
    # Full run (per model)
    # ------------------------------------------------------------------

    def _run_all_models(
        self,
        models: list[ModelConfig],
        game_logs: list[AIWolfGameLog],
        run_dir: RunDirectory,
    ) -> dict[str, ProcessingResult]:
        """全モデルを実行し、モデルID -> ProcessingResult の辞書を返す."""
        if self.processing_config.parallel_models and len(models) > 1:
            return self._run_all_models_parallel(models, game_logs, run_dir)
        return self._run_all_models_sequential(models, game_logs, run_dir)

    def _run_all_models_sequential(
        self,
        models: list[ModelConfig],
        game_logs: list[AIWolfGameLog],
        run_dir: RunDirectory,
    ) -> dict[str, ProcessingResult]:
        results: dict[str, ProcessingResult] = {}
        for model_config in models:
            results[model_config.id] = self._run_one_model(
                model_config, game_logs, run_dir
            )
        return results

    def _run_all_models_parallel(
        self,
        models: list[ModelConfig],
        game_logs: list[AIWolfGameLog],
        run_dir: RunDirectory,
    ) -> dict[str, ProcessingResult]:
        results: dict[str, ProcessingResult] = {}
        with ThreadPoolExecutor(max_workers=len(models)) as pool:
            future_to_model = {
                pool.submit(self._run_one_model, m, game_logs, run_dir): m
                for m in models
            }
            for future in as_completed(future_to_model):
                m = future_to_model[future]
                try:
                    results[m.id] = future.result()
                except ConfigurationError:
                    # 設定ミスは伝播。pending な他モデルもキャンセル相当として扱う。
                    raise
                except Exception as e:
                    logger.error(f"Model '{m.id}' run crashed: {e}", exc_info=True)
                    results[m.id] = ProcessingResult(
                        total=len(game_logs),
                        completed=0,
                        failed=len(game_logs),
                    )
        return results

    def _run_one_model(
        self,
        model_config: ModelConfig,
        game_logs: list[AIWolfGameLog],
        run_dir: RunDirectory,
    ) -> ProcessingResult:
        logger.info(f"=== Running model '{model_config.id}' ===")
        model_output_dir = run_dir.model_dir(model_config.id)

        if self.processing_config.use_batch_api:
            result = self._run_one_model_batch(
                model_config, game_logs, model_output_dir
            )
        else:
            result = self._execute_parallel_games(
                model_config, game_logs, model_output_dir
            )

        if result.completed > 0:
            self._generate_team_aggregation(
                model_config, result.evaluation_results, model_output_dir
            )
        logger.info(
            f"=== Model '{model_config.id}' done: "
            f"{result.completed}/{result.total} success ==="
        )
        return result

    def _run_one_model_batch(
        self,
        model_config: ModelConfig,
        game_logs: list[AIWolfGameLog],
        model_output_dir: Path,
    ) -> ProcessingResult:
        """バッチAPIモードで1モデル分を実行."""
        from src.processor.batch_orchestrator import BatchOrchestrator

        result = ProcessingResult(total=len(game_logs))
        try:
            orch = BatchOrchestrator(self.config, model_config)
        except ConfigurationError:
            raise
        except Exception as e:
            logger.error(
                f"[{model_config.id}] failed to construct BatchOrchestrator: {e}",
                exc_info=True,
            )
            result.failed = len(game_logs)
            return result

        if not orch.can_use_batch():
            logger.info(
                f"[{model_config.id}] provider does not support batch API; "
                f"falling back to synchronous path"
            )
            return self._execute_parallel_games(
                model_config, game_logs, model_output_dir
            )

        try:
            completed, failed, evaluation_dicts = orch.run(
                game_logs,
                model_output_dir,
                poll_interval_seconds=self.processing_config.batch_poll_interval_seconds,
                max_wait_seconds=self.processing_config.batch_max_wait_seconds,
            )
            result.completed = completed
            result.failed = failed
            result.evaluation_results.extend(evaluation_dicts)
        except Exception as e:
            logger.error(
                f"[{model_config.id}] batch run failed: {e}", exc_info=True
            )
            result.failed = len(game_logs) - result.completed
        return result

    def _execute_parallel_games(
        self,
        model_config: ModelConfig,
        game_logs: list[AIWolfGameLog],
        output_dir: Path,
    ) -> ProcessingResult:
        result = ProcessingResult(total=len(game_logs))

        # 'spawn' を明示することで、parallel_models=True 時に ThreadPool 内から
        # この ProcessPool が起動される場合の fork() 起因のデッドロックを避ける。
        # 単独実行時もインポート再評価のコスト以外は安全側に倒れる。
        with ProcessPoolExecutor(
            max_workers=self.processing_config.max_workers,
            mp_context=mp.get_context("spawn"),
        ) as executor:
            futures = [
                (
                    executor.submit(
                        BatchProcessor._process_single_game_worker,
                        game_log,
                        self.config,
                        model_config,
                        output_dir,
                    ),
                    game_log,
                )
                for game_log in game_logs
            ]

            for future, game_log in futures:
                try:
                    success, evaluation_dict = future.result()
                    if success and evaluation_dict:
                        result.completed += 1
                        result.evaluation_results.append(evaluation_dict)
                        logger.info(
                            f"[{model_config.id}] "
                            f"{GameProcessor.SUCCESS_INDICATOR} Completed: "
                            f"{game_log.game_id}"
                        )
                    else:
                        result.failed += 1
                        logger.error(
                            f"[{model_config.id}] "
                            f"{GameProcessor.FAILURE_INDICATOR} Failed: "
                            f"{game_log.game_id}"
                        )
                except Exception as e:
                    result.failed += 1
                    logger.error(
                        f"[{model_config.id}] "
                        f"{GameProcessor.FAILURE_INDICATOR} Error processing "
                        f"{game_log.game_id}: {e}"
                    )

        return result

    @staticmethod
    def _process_single_game_worker(
        game_log: AIWolfGameLog,
        config: dict[str, Any],
        model_config: ModelConfig,
        output_dir: Path,
    ) -> tuple[bool, dict | None]:
        """ProcessPoolExecutor で実行される単一ゲーム処理."""
        processor = GameProcessor(config, model_config)
        return processor.process(game_log, output_dir)

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _generate_team_aggregation(
        self,
        model_config: ModelConfig,
        evaluation_results: list[dict],
        model_output_dir: Path,
    ) -> None:
        self._aggregation_service.generate_and_save(
            evaluation_results, model_output_dir, model_id=model_config.id
        )

    def _regenerate_aggregation_for_model_dir(self, model_dir: Path) -> None:
        self._aggregation_service.regenerate_for_model_dir(model_dir)
