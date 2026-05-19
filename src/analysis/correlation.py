"""メタ情報 × LLM-Judge rank の相関分析（バイアス監査用）."""

from __future__ import annotations

import pandas as pd
from scipy import stats


NUMERIC_META = ["vote_rate", "survival_rate", "talk_rate"]
CATEGORICAL_META = ["death_cause", "won", "role"]


def join_rank_with_meta(
    rankings: pd.DataFrame,
    meta: pd.DataFrame,
) -> pd.DataFrame:
    """rankings と meta を (ulid, player_name) で結合.

    結合キー: (ulid, player_name)。team_name の表記揺れに依存しない。
    """
    if meta.empty:
        return rankings.assign(
            vote_rate=None,
            survival_rate=None,
            talk_rate=None,
            death_cause=None,
            won=None,
            role=None,
        )
    cols = ["ulid", "player_name", "vote_rate", "survival_rate", "talk_rate",
            "death_cause", "won", "role"]
    meta_slim = meta[cols].drop_duplicates(subset=["ulid", "player_name"])
    merged = rankings.merge(meta_slim, on=["ulid", "player_name"], how="left")
    return merged


def numeric_meta_correlation(merged: pd.DataFrame) -> pd.DataFrame:
    """数値メタ情報（vote_rate / survival_rate / talk_rate）と rank の Spearman 相関.

    モデル × 基準 × メタ指標ごと。

    Returns:
        DataFrame: columns: model, criterion, meta, spearman_r, n
    """
    rows: list[dict] = []
    for (model, crit), sub in merged.groupby(["model", "criterion"]):
        for meta in NUMERIC_META:
            paired = sub[[meta, "ranking"]].dropna()
            if len(paired) < 3:
                continue
            r, _ = stats.spearmanr(paired[meta], paired["ranking"])
            rows.append({
                "model": model,
                "criterion": crit,
                "meta": meta,
                "spearman_r": float(r) if r == r else None,
                "n": int(len(paired)),
            })
    return pd.DataFrame(rows)


def avg_rank_by_player(merged: pd.DataFrame) -> pd.DataFrame:
    """各プレイヤー単位で「全基準平均 rank」を計算（モデル別）.

    プレイヤー単位の bias 監査用（票や死因と avg_rank の関係を見る）。

    Returns:
        DataFrame: ulid, team, player_name, role, model, avg_rank,
                   vote_rate, survival_rate, talk_rate, death_cause, won
    """
    grp_keys = ["ulid", "team", "player_name", "role", "model",
                "vote_rate", "survival_rate", "talk_rate", "death_cause", "won"]
    avg = (
        merged.groupby(grp_keys, dropna=False)["ranking"]
        .mean()
        .reset_index(name="avg_rank")
    )
    return avg


def avg_rank_by_category(
    avg_by_player: pd.DataFrame,
    category: str,
) -> pd.DataFrame:
    """カテゴリ別（死因/勝敗/役職）の avg_rank 統計.

    Returns:
        DataFrame: columns: model, <category>, mean_rank, n
    """
    if category not in avg_by_player.columns:
        return pd.DataFrame()
    return (
        avg_by_player.groupby(["model", category], dropna=False)["avg_rank"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_rank", "count": "n"})
    )


def numeric_meta_correlation_avg(avg_by_player: pd.DataFrame) -> pd.DataFrame:
    """avg_rank 単位での数値メタ相関（B5 構造指標相当）.

    Returns:
        DataFrame: columns: model, meta, spearman_r, n
    """
    rows: list[dict] = []
    for model, sub in avg_by_player.groupby("model"):
        for meta in NUMERIC_META:
            paired = sub[[meta, "avg_rank"]].dropna()
            if len(paired) < 3:
                continue
            r, _ = stats.spearmanr(paired[meta], paired["avg_rank"])
            rows.append({
                "model": model,
                "meta": meta,
                "spearman_r": float(r) if r == r else None,
                "n": int(len(paired)),
            })
    return pd.DataFrame(rows)
