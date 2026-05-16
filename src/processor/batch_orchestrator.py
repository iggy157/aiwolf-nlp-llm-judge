"""バッチAPIモード用オーケストレータ.

全ゲーム × 全評価基準のリクエストをまとめてプロバイダのバッチAPIに投入し、
完了後に結果を回収して、既存の同期パスと同じ出力フォーマット（`*_result.json`
+ `team_aggregation.{json,csv}`）に整形する。

ローカル推論サーバ（openai_compatible）は build_batch_client が None を返すので
本オーケストレータは使われず、同期パスにフォールバックする想定。
"""

import json
import logging
from pathlib import Path
from typing import Any

from src.aiwolf_log.game_log import AIWolfGameLog
from src.evaluation.loaders.criteria_loader import CriteriaLoader
from src.evaluation.loaders.settings_loader import SettingsLoader
from src.evaluation.models.config import EvaluationConfig
from src.evaluation.models.criteria import EvaluationCriteria
from src.evaluation.models.llm_response import EvaluationLLMResponse
from src.evaluation.models.result import CriteriaEvaluationResult, EvaluationResult
from src.game.detector import GameDetector
from src.game.models import GameInfo
from src.llm.batch import BatchClient, BatchRequest, BatchResult
from src.llm.client import ModelConfig, PromptTemplates
from src.llm.factory import build_batch_client
from src.llm.formatter import GameLogFormatter
from src.processor.models.exceptions import EvaluationExecutionError
from src.processor.pipeline.result_writing import ResultWritingService

logger = logging.getLogger(__name__)


CUSTOM_ID_SEPARATOR = "::"


class BatchOrchestrator:
    """バッチAPIモードのオーケストレータ.

    `BatchProcessor` から1モデル単位で呼ばれることを想定:
        orch = BatchOrchestrator(config, model_config)
        result = orch.run(game_logs, output_dir)
    戻り値の構造は同期パス（`_run_one_model`）の `ProcessingResult` と同じ。
    """

    def __init__(self, config: dict[str, Any], model_config: ModelConfig) -> None:
        self.config = config
        self.model_config = model_config
        self.settings_path = Path(config["settings_path"])
        self.result_service = ResultWritingService()

        # プロンプトテンプレート
        self._templates = self._load_templates()
        # 評価設定
        self._evaluation_config: EvaluationConfig = CriteriaLoader.load_evaluation_config(
            SettingsLoader.get_evaluation_criteria_path(self.settings_path)
        )

    def can_use_batch(self) -> bool:
        """このモデルのバッチクライアントが構築できる（=バッチAPI使用可能）か."""
        client = build_batch_client(self.model_config)
        return client is not None

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------

    def run(
        self,
        game_logs: list[AIWolfGameLog],
        output_dir: Path,
        poll_interval_seconds: float = 60.0,
        max_wait_seconds: float = 86400.0,
    ) -> tuple[int, int, list[dict]]:
        """バッチ実行を完遂し (completed_games, failed_games, evaluation_dicts) を返す."""
        batch_client = build_batch_client(self.model_config)
        if batch_client is None:
            raise EvaluationExecutionError(
                f"{self.model_config.id}: provider does not support batch API"
            )

        # 1. 全ゲームの前処理（CSV解析、フォーマット、game_info検出、character_info取得）
        game_contexts = self._prepare_game_contexts(game_logs)
        if not game_contexts:
            return 0, 0, []

        # 2. 全 (game, criteria) のバッチリクエストを構築
        batch_requests = self._build_batch_requests(game_contexts)
        logger.info(
            f"[{self.model_config.id}] built {len(batch_requests)} batch requests "
            f"for {len(game_contexts)} games"
        )

        # 3. バッチ投入＆ポーリング＆結果取得
        batch_results = batch_client.submit_and_wait(
            requests=batch_requests,
            templates=self._templates,
            output_structure=EvaluationLLMResponse,
            poll_interval_seconds=poll_interval_seconds,
            max_wait_seconds=max_wait_seconds,
        )

        # 4. 結果を game 毎に集約し、ファイル保存
        return self._assemble_and_save(game_contexts, batch_results, output_dir)

    # ------------------------------------------------------------------
    # Preparation
    # ------------------------------------------------------------------

    def _load_templates(self) -> PromptTemplates:
        import yaml

        prompt_yml_path = Path(self.config["llm"]["prompt_yml"])
        with prompt_yml_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        system_template = data.get("developer") or data.get("system")
        user_template = data.get("user")
        if not system_template or not user_template:
            raise EvaluationExecutionError(
                "prompts.yaml must contain 'developer' (or 'system') and 'user'"
            )
        return PromptTemplates(system=system_template, user=user_template)

    def _prepare_game_contexts(
        self, game_logs: list[AIWolfGameLog]
    ) -> list[dict[str, Any]]:
        """各ゲームを評価可能な状態まで処理した context dict のリストを返す."""
        contexts: list[dict[str, Any]] = []
        for game_log in game_logs:
            try:
                game_info = GameDetector.detect_game_format(
                    game_log.log_path, self.settings_path
                )
                criteria_for_game = self._evaluation_config.get_criteria_for_game(
                    game_info
                )
                if not criteria_for_game:
                    logger.warning(
                        f"[{self.model_config.id}] no criteria for {game_log.game_id}; "
                        "skipping"
                    )
                    continue
                formatter = GameLogFormatter(game_log, self.config)
                formatted_data = formatter.convert_to_jsonl(game_info.game_format)
                character_info_str = self._build_character_info_string(game_log)
                contexts.append(
                    {
                        "game_log": game_log,
                        "game_info": game_info,
                        "criteria": criteria_for_game,
                        "formatted_data": formatted_data,
                        "character_info": character_info_str,
                        "agent_to_team": game_log.get_agent_to_team_mapping(),
                    }
                )
            except Exception as e:
                logger.error(
                    f"[{self.model_config.id}] failed to prepare {game_log.game_id}: "
                    f"{e}",
                    exc_info=True,
                )
        return contexts

    @staticmethod
    def _build_character_info_string(game_log: AIWolfGameLog) -> str:
        """同期パス（LogFormattingService.get_character_info）と同形式で生成."""
        try:
            profiles = game_log.get_character_info()
        except Exception:
            return ""
        return "\n".join(f"- {name}: {profile}" for name, profile in profiles.items())

    # ------------------------------------------------------------------
    # Batch request building
    # ------------------------------------------------------------------

    def _build_batch_requests(
        self, game_contexts: list[dict[str, Any]]
    ) -> list[BatchRequest]:
        requests: list[BatchRequest] = []
        for ctx in game_contexts:
            game_id = ctx["game_log"].game_id
            log_json = json.dumps(ctx["formatted_data"], ensure_ascii=False)
            for criteria in ctx["criteria"]:
                requests.append(
                    BatchRequest(
                        custom_id=self._make_custom_id(game_id, criteria.name),
                        criteria_description=criteria.description,
                        character_info=ctx["character_info"],
                        log_json=log_json,
                    )
                )
        return requests

    @staticmethod
    def _make_custom_id(game_id: str, criteria_name: str) -> str:
        return f"{game_id}{CUSTOM_ID_SEPARATOR}{criteria_name}"

    @staticmethod
    def _parse_custom_id(custom_id: str) -> tuple[str, str]:
        parts = custom_id.split(CUSTOM_ID_SEPARATOR, 1)
        if len(parts) != 2:
            return custom_id, ""
        return parts[0], parts[1]

    # ------------------------------------------------------------------
    # Result assembly
    # ------------------------------------------------------------------

    def _assemble_and_save(
        self,
        game_contexts: list[dict[str, Any]],
        batch_results: list[BatchResult],
        output_dir: Path,
    ) -> tuple[int, int, list[dict]]:
        # custom_id -> BatchResult
        result_by_id: dict[str, BatchResult] = {r.custom_id: r for r in batch_results}

        ctx_by_game_id: dict[str, dict[str, Any]] = {
            ctx["game_log"].game_id: ctx for ctx in game_contexts
        }

        completed = 0
        failed = 0
        evaluation_dicts: list[dict] = []

        for game_id, ctx in ctx_by_game_id.items():
            game_log: AIWolfGameLog = ctx["game_log"]
            game_info: GameInfo = ctx["game_info"]
            criteria_for_game: list[EvaluationCriteria] = ctx["criteria"]
            agent_to_team: dict[str, str] = ctx["agent_to_team"]

            try:
                game_result = self._assemble_one_game(
                    game_id, criteria_for_game, result_by_id, agent_to_team
                )
                self.result_service.save_results(
                    game_log, game_info, game_result, output_dir
                )
                evaluation_dicts.append(game_result.to_dict())
                completed += 1
                logger.info(
                    f"[{self.model_config.id}] ✓ Completed (batch): {game_id}"
                )
            except Exception as e:
                failed += 1
                logger.error(
                    f"[{self.model_config.id}] ✗ Failed to assemble {game_id}: {e}",
                    exc_info=True,
                )

        return completed, failed, evaluation_dicts

    def _assemble_one_game(
        self,
        game_id: str,
        criteria_for_game: list[EvaluationCriteria],
        result_by_id: dict[str, BatchResult],
        agent_to_team: dict[str, str],
    ) -> EvaluationResult:
        evaluation_result = EvaluationResult()
        missing: list[str] = []
        errors: list[str] = []
        for criteria in criteria_for_game:
            cid = self._make_custom_id(game_id, criteria.name)
            r = result_by_id.get(cid)
            if r is None:
                missing.append(criteria.name)
                continue
            if not r.success or r.response is None:
                errors.append(f"{criteria.name}: {r.error}")
                continue
            criteria_result = CriteriaEvaluationResult.from_llm_response(
                criteria.name, r.response, agent_to_team
            )
            evaluation_result.append(criteria_result)

        if missing or errors:
            raise EvaluationExecutionError(
                f"game {game_id} incomplete: "
                f"missing={missing}, errors={errors}"
            )
        return evaluation_result
