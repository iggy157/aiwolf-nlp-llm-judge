"""分析 CLI のエントリーポイント.

Usage:
    uv run python -m src.analysis <input_dir> [--input-data data/input] [--no-plots]
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from src.analysis.runner import run_analysis


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "LLM-Judge 評価結果のクロスモデル分析。"
            "output/result/<name>/ にユーザーが集めたモデル別フォルダを入力に、"
            "output/analysis/<name>/<timestamp>/ に分析結果を出力する。"
        )
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="モデル別フォルダを含む入力ディレクトリ（例: output/result/main_run）",
    )
    parser.add_argument(
        "--input-data",
        type=Path,
        default=Path("data/input"),
        help="ゲームログ/JSONの所在（メタ情報抽出用、デフォルト: data/input）",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/output/analysis"),
        help="分析結果の出力ルート（デフォルト: data/output/analysis）",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="プロット生成をスキップ（CSV と Markdown のみ）",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.input_dir.is_dir():
        print(f"入力ディレクトリが見つかりません: {args.input_dir}", file=sys.stderr)
        return 1

    name = args.input_dir.name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = args.output_root / name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"分析開始: input={args.input_dir} output={output_dir}")
    run_analysis(
        input_dir=args.input_dir,
        input_data_dir=args.input_data,
        output_dir=output_dir,
        make_plots=not args.no_plots,
    )
    logging.info(f"分析完了: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
