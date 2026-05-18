import argparse
import logging
from pathlib import Path

import yaml

from src.processor.batch_processor import BatchProcessor


def setup_logging() -> None:
    """ロギングの設定."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def _split_csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def main() -> None:
    """メイン処理."""
    parser = argparse.ArgumentParser(description="AIWolf NLP LLM Judge")

    parser.add_argument(
        "-c", "--config", type=Path, required=True, help="設定ファイルのパス"
    )
    parser.add_argument("--debug", action="store_true", help="デバッグモードで実行")
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="実行対象モデルIDをカンマ区切りで指定（例: gpt-4o,claude-sonnet-4-5）。"
        "未指定なら llm.models 全件を実行",
    )
    parser.add_argument(
        "--skip-dry-run",
        action="store_true",
        help="本実行前の試運転をスキップ（settings.yamlのdry_runより優先）",
    )
    parser.add_argument(
        "--dry-run-only",
        action="store_true",
        help="試運転のみ実行し、本実行は行わない",
    )
    parser.add_argument(
        "--use-batch",
        action="store_true",
        help="本実行にプロバイダのBatch APIを使用（24h SLA、50%%引）。"
        "settings.yamlのuse_batch_apiより優先",
    )
    parser.add_argument(
        "--no-batch",
        action="store_true",
        help="settings.yamlで use_batch_api: true でも同期実行を強制",
    )
    parser.add_argument(
        "--parallel-models",
        action="store_true",
        help="複数モデルを同時並行で実行（settings.yamlのparallel_modelsより優先）。"
        "closed系のクラウドAPIプロバイダ間ではレートリミットが独立なので安全だが、"
        "ローカル推論サーバ（openai_compatible）を含む場合はVRAM競合に注意",
    )
    parser.add_argument(
        "--no-parallel-models",
        action="store_true",
        help="settings.yamlで parallel_models: true でも逐次実行を強制",
    )

    parser.add_argument(
        "--regenerate-aggregation",
        action="store_true",
        help="既存の評価結果JSONからチーム集計を再生成",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="集計再生成の対象 <output_root>/<timestamp> ディレクトリ。"
        "未指定なら最新のタイムスタンプディレクトリを使用",
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.config.is_file():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {args.config}")

    try:
        with args.config.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            logging.info(f"設定ファイルを読み込みました: {args.config}")
    except Exception as e:
        raise RuntimeError(f"設定ファイルの読み込みに失敗しました: {e}")

    config["settings_path"] = str(args.config)

    # CLIフラグで dry_run / use_batch_api を上書き
    if args.skip_dry_run:
        config.setdefault("processing", {})["dry_run"] = False
    if args.use_batch and args.no_batch:
        raise ValueError("Cannot specify both --use-batch and --no-batch")
    if args.use_batch:
        config.setdefault("processing", {})["use_batch_api"] = True
    if args.no_batch:
        config.setdefault("processing", {})["use_batch_api"] = False
    if args.parallel_models and args.no_parallel_models:
        raise ValueError(
            "Cannot specify both --parallel-models and --no-parallel-models"
        )
    if args.parallel_models:
        config.setdefault("processing", {})["parallel_models"] = True
    if args.no_parallel_models:
        config.setdefault("processing", {})["parallel_models"] = False

    model_ids_filter = _split_csv(args.models)
    processor = BatchProcessor(config, model_ids_filter=model_ids_filter)

    if args.regenerate_aggregation:
        logging.info("既存の評価結果から集計を再生成します...")
        processor.regenerate_aggregation_only(run_dir=args.run_dir)
        logging.info("集計の再生成が完了しました")
        return

    result = processor.process_all_games(dry_run_only=args.dry_run_only)
    if result.total > 0:
        logging.info(
            f"処理完了 - 成功: {result.completed}/{result.total}, "
            f"成功率: {result.success_rate:.2%}"
        )
    else:
        logging.info("処理対象がありませんでした")


if __name__ == "__main__":
    setup_logging()
    try:
        main()
    except KeyboardInterrupt:
        logging.info("処理が中断されました")
    except Exception as e:
        logging.error(f"エラーが発生しました: {e}")
        exit(1)
