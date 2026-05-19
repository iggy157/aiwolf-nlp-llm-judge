"""LLM-Judge 評価結果の分析モジュール.

入力: output/result/<name>/ にユーザーが手動で集めたモデル別フォルダ
出力: output/analysis/<name>/<timestamp>/ 配下に分析結果（CSV / PNG / Markdown）

使い方:
    uv run python -m src.analysis output/result/<name>
"""
