"""モデル間比較の分析（相関・同順位率・乖離大の事例抽出）."""

from __future__ import annotations

import itertools
from typing import Iterable

import pandas as pd
from scipy import stats


def pairwise_rank_correlation(rankings: pd.DataFrame) -> pd.DataFrame:
    """各モデルペア × 各基準について、プレイヤー単位の rank の Spearman 相関を計算.

    プレイヤー単位は (ulid, player_name) の組で識別する。

    Returns:
        DataFrame with columns: criterion, model_a, model_b, spearman_r, n
    """
    rows: list[dict] = []
    pivot = rankings.pivot_table(
        index=["criterion", "ulid", "player_name"],
        columns="model",
        values="ranking",
        aggfunc="first",
    )
    pivot = pivot.reset_index()
    models = [c for c in pivot.columns if c not in ("criterion", "ulid", "player_name")]
    for crit, sub in pivot.groupby("criterion"):
        for a, b in itertools.combinations(models, 2):
            paired = sub[[a, b]].dropna()
            if len(paired) < 3:
                continue
            r, _ = stats.spearmanr(paired[a], paired[b])
            rows.append({
                "criterion": crit,
                "model_a": a,
                "model_b": b,
                "spearman_r": float(r) if r == r else None,  # NaN チェック
                "n": int(len(paired)),
            })
    return pd.DataFrame(rows)


def tie_rates(rankings: pd.DataFrame) -> pd.DataFrame:
    """各モデル × 各基準で、同順位を含むゲームの比率を計算.

    Returns:
        DataFrame: index=model, columns=criterion, values=tie_rate (0〜1)
    """
    # (model, criterion, ulid) ごとに rank のリストを取り、重複があれば tie=1
    grp = (
        rankings.groupby(["model", "criterion", "ulid"])["ranking"]
        .apply(lambda s: int(len(s) != len(set(s))))
        .reset_index(name="has_tie")
    )
    rate = grp.groupby(["model", "criterion"])["has_tie"].mean().unstack(fill_value=0.0)
    rate.columns.name = None
    return rate


def divergent_cases(rankings: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """team × criterion × ulid 単位で、モデル間 rank の max-min が大きい事例を抽出.

    Returns:
        DataFrame with columns: ulid, team, player_name, criterion, <model>...,
                                max_min_diff, n_models
    """
    pivot = rankings.pivot_table(
        index=["ulid", "team", "player_name", "criterion"],
        columns="model",
        values="ranking",
        aggfunc="first",
    ).reset_index()
    models = [c for c in pivot.columns
              if c not in ("ulid", "team", "player_name", "criterion")]
    rank_cols = pivot[models]
    pivot["max_min_diff"] = rank_cols.max(axis=1) - rank_cols.min(axis=1)
    pivot["n_models"] = rank_cols.notna().sum(axis=1)
    pivot = pivot.dropna(subset=["max_min_diff"])
    pivot = pivot[pivot["n_models"] >= 2]
    pivot = pivot.sort_values("max_min_diff", ascending=False).head(top_n)
    return pivot


def team_avg_rank(rankings: pd.DataFrame) -> pd.DataFrame:
    """チーム × 基準 × モデルの avg_rank（ゲーム平均）を返す.

    Returns:
        DataFrame: long形式  columns: team, criterion, model, avg_rank, n_games
    """
    grp = (
        rankings.groupby(["team", "criterion", "model"])["ranking"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "avg_rank", "count": "n_games"})
    )
    return grp


def criteria_correlation(rankings: pd.DataFrame) -> pd.DataFrame:
    """各モデル内で「基準どうしの rank 相関」を計算.

    Returns:
        DataFrame: long形式  columns: model, criterion_a, criterion_b, spearman_r, n
    """
    rows: list[dict] = []
    pivot = rankings.pivot_table(
        index=["model", "ulid", "player_name"],
        columns="criterion",
        values="ranking",
        aggfunc="first",
    ).reset_index()
    criteria = [c for c in pivot.columns if c not in ("model", "ulid", "player_name")]
    for model, sub in pivot.groupby("model"):
        for a, b in itertools.combinations(criteria, 2):
            paired = sub[[a, b]].dropna()
            if len(paired) < 3:
                continue
            r, _ = stats.spearmanr(paired[a], paired[b])
            rows.append({
                "model": model,
                "criterion_a": a,
                "criterion_b": b,
                "spearman_r": float(r) if r == r else None,
                "n": int(len(paired)),
            })
    return pd.DataFrame(rows)
