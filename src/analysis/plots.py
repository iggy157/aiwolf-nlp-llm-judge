"""matplotlib / seaborn によるプロット生成."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # ヘッドレス
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# 日本語フォント対応（フォールバック）。フォントがなくても警告のみで継続。
plt.rcParams["font.family"] = ["DejaVu Sans", "Noto Sans CJK JP", "IPAGothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid")


def plot_rank_correlation_heatmap(
    pair_corr: pd.DataFrame,
    output_path: Path,
) -> None:
    """モデル間相関ヒートマップ（criterion ごとにパネル）."""
    if pair_corr.empty:
        logger.info("プロットスキップ: pair_corr 空")
        return
    criteria = sorted(pair_corr["criterion"].unique())
    n = len(criteria)
    if n == 0:
        return
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    for ax in axes.flatten()[n:]:
        ax.axis("off")
    for i, crit in enumerate(criteria):
        ax = axes[i // cols][i % cols]
        sub = pair_corr[pair_corr["criterion"] == crit]
        models = sorted(set(sub["model_a"]) | set(sub["model_b"]))
        mat = pd.DataFrame(np.nan, index=models, columns=models)
        for _, row in sub.iterrows():
            mat.loc[row["model_a"], row["model_b"]] = row["spearman_r"]
            mat.loc[row["model_b"], row["model_a"]] = row["spearman_r"]
        for m in models:
            mat.loc[m, m] = 1.0
        sns.heatmap(
            mat.astype(float), annot=True, fmt=".2f", cmap="coolwarm",
            vmin=-1, vmax=1, ax=ax, cbar=False,
        )
        ax.set_title(crit, fontsize=10)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.tick_params(axis="y", rotation=0, labelsize=8)
    fig.suptitle("Model pair rank correlation (Spearman) per criterion", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"saved: {output_path}")


def plot_tie_rates_bar(tie_rates: pd.DataFrame, output_path: Path) -> None:
    """同順位率の棒グラフ（モデル × 基準）."""
    if tie_rates.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4 + 0.5 * len(tie_rates)))
    tie_rates.plot.barh(ax=ax)
    ax.set_xlabel("Tie usage rate (games with ties / total games)")
    ax.set_xlim(0, 1.0)
    ax.set_title("Same-rank (tie) usage rate per criterion × model")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"saved: {output_path}")


def plot_meta_scatter_panel(
    avg_by_player: pd.DataFrame,
    output_path: Path,
) -> None:
    """メタ指標 vs avg_rank 散布図のパネル（モデル × 指標）."""
    if avg_by_player.empty:
        return
    metas = ["vote_rate", "survival_rate", "talk_rate"]
    models = sorted(avg_by_player["model"].dropna().unique())
    if not models:
        return
    fig, axes = plt.subplots(
        len(models), len(metas), figsize=(4 * len(metas), 3.5 * len(models)),
        squeeze=False,
    )
    for i, model in enumerate(models):
        sub = avg_by_player[avg_by_player["model"] == model]
        for j, meta in enumerate(metas):
            ax = axes[i][j]
            paired = sub[[meta, "avg_rank"]].dropna()
            if paired.empty:
                ax.axis("off")
                continue
            sns.regplot(
                data=paired, x=meta, y="avg_rank", ax=ax,
                scatter_kws={"s": 14, "alpha": 0.55},
                line_kws={"color": "crimson"},
            )
            from scipy import stats
            r, _ = stats.spearmanr(paired[meta], paired["avg_rank"])
            ax.set_title(f"{model} | {meta}  r={r:.2f}", fontsize=9)
            ax.set_ylabel("avg_rank" if j == 0 else "")
    fig.suptitle("Meta info vs avg_rank (lower rank = better)", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"saved: {output_path}")


def plot_team_avg_rank(
    team_avg: pd.DataFrame,
    output_path: Path,
) -> None:
    """チーム × モデルの avg_rank ヒートマップ（基準は平均集約）."""
    if team_avg.empty:
        return
    # 各 team × model で全基準平均
    grp = team_avg.groupby(["team", "model"])["avg_rank"].mean().unstack("model")
    if grp.empty:
        return
    # 全モデル平均で並べ替え
    grp = grp.assign(_overall=grp.mean(axis=1)).sort_values("_overall").drop(columns="_overall")
    fig, ax = plt.subplots(figsize=(2 + 1.2 * len(grp.columns), 0.32 * len(grp) + 1.5))
    sns.heatmap(grp, annot=True, fmt=".2f", cmap="RdYlGn_r", ax=ax, vmin=1, vmax=5)
    ax.set_title("Team avg_rank per model (avg over criteria)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"saved: {output_path}")


def plot_criteria_correlation(
    crit_corr: pd.DataFrame,
    output_path: Path,
) -> None:
    """同一モデル内の基準間相関ヒートマップ（モデル別パネル）."""
    if crit_corr.empty:
        return
    models = sorted(crit_corr["model"].unique())
    n = len(models)
    if n == 0:
        return
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    for ax in axes.flatten()[n:]:
        ax.axis("off")
    for i, model in enumerate(models):
        ax = axes[i // cols][i % cols]
        sub = crit_corr[crit_corr["model"] == model]
        criteria = sorted(set(sub["criterion_a"]) | set(sub["criterion_b"]))
        mat = pd.DataFrame(np.nan, index=criteria, columns=criteria)
        for _, row in sub.iterrows():
            mat.loc[row["criterion_a"], row["criterion_b"]] = row["spearman_r"]
            mat.loc[row["criterion_b"], row["criterion_a"]] = row["spearman_r"]
        for c in criteria:
            mat.loc[c, c] = 1.0
        sns.heatmap(
            mat.astype(float), annot=True, fmt=".2f", cmap="coolwarm",
            vmin=-1, vmax=1, ax=ax, cbar=False,
        )
        ax.set_title(model, fontsize=10)
        ax.tick_params(axis="x", rotation=30, labelsize=7)
        ax.tick_params(axis="y", rotation=0, labelsize=7)
    fig.suptitle("Criteria correlation per model (Spearman)", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"saved: {output_path}")
