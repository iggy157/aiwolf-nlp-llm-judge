"""モデル別評価結果と入力ログのローダー."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def discover_models(input_dir: Path) -> list[str]:
    """input_dir 直下のモデル別フォルダ名を発見.

    Args:
        input_dir: output/result/<name>/

    Returns:
        モデルID（フォルダ名）のリスト。`*_result.json` が1つ以上あるフォルダのみ採用。
    """
    models: list[str] = []
    for sub in sorted(input_dir.iterdir()):
        if not sub.is_dir():
            continue
        if not list(sub.glob("*_result.json")):
            logger.warning(f"スキップ（result.json が見つからない）: {sub}")
            continue
        models.append(sub.name)
    if not models:
        raise ValueError(f"モデルフォルダが {input_dir} に見つかりません")
    logger.info(f"検出したモデル: {models}")
    return models


def load_rankings(input_dir: Path, models: list[str]) -> pd.DataFrame:
    """全モデル・全ゲーム・全基準・全プレイヤーの順位をフラットなDataFrameで返す.

    Returns:
        DataFrame with columns:
          model, ulid, criterion, team, player_name, ranking, reasoning
    """
    rows: list[dict[str, Any]] = []
    for model in models:
        model_dir = input_dir / model
        for result_path in sorted(model_dir.glob("*_result.json")):
            ulid = result_path.stem.removesuffix("_result")
            try:
                data = json.loads(result_path.read_text())
            except json.JSONDecodeError as e:
                logger.warning(f"JSON 解析失敗 {result_path}: {e}")
                continue
            evaluations = data.get("evaluations", data)
            if not isinstance(evaluations, dict):
                continue
            for crit_name, crit_data in evaluations.items():
                if not isinstance(crit_data, dict):
                    continue
                rankings = crit_data.get("rankings", [])
                for r in rankings:
                    rows.append({
                        "model": model,
                        "ulid": ulid,
                        "criterion": crit_name,
                        "team": r.get("team", ""),
                        "player_name": r.get("player_name", ""),
                        "ranking": r.get("ranking"),
                        "reasoning": r.get("reasoning", ""),
                    })
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"順位データが見つかりません: {input_dir}")
    logger.info(f"順位データ: {len(df)} rows, {df['model'].nunique()} models, "
                f"{df['ulid'].nunique()} games, {df['criterion'].nunique()} criteria")
    return df


def map_ulid_to_logfile(input_data_dir: Path) -> dict[str, str]:
    """JSON の game_id (ULID) からログファイル名 (gameN) へのマッピングを作る."""
    mapping: dict[str, str] = {}
    json_dir = input_data_dir / "json"
    for json_path in json_dir.glob("*.json"):
        try:
            data = json.loads(json_path.read_text())
        except json.JSONDecodeError:
            continue
        gid = data.get("game_id")
        if gid:
            mapping[gid] = json_path.stem
    if not mapping:
        logger.warning(f"ULID マッピングが空: {input_data_dir}")
    return mapping
