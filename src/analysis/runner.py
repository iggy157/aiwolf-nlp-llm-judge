"""分析パイプラインのオーケストレータ."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.analysis import correlation as corr_mod
from src.analysis import cross_model as cm
from src.analysis import loader, meta as meta_mod, plots
from src.analysis.summary import write_summary

logger = logging.getLogger(__name__)


def run_analysis(
    input_dir: Path,
    input_data_dir: Path,
    output_dir: Path,
    make_plots: bool = True,
) -> None:
    """全分析パイプラインを実行し、output_dir に結果を保存."""
    # --- 1. ロード ---
    models = loader.discover_models(input_dir)
    rankings = loader.load_rankings(input_dir, models)
    ulid_map = loader.map_ulid_to_logfile(input_data_dir)
    ulids = sorted(rankings["ulid"].unique())
    meta = meta_mod.extract_meta(input_data_dir, ulid_map, ulids)

    # 保存先のサブディレクトリ
    cross_dir = output_dir / "cross_model"
    meta_dir = output_dir / "meta_correlation"
    plot_dir = output_dir / "plots"
    for d in (cross_dir, meta_dir, plot_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 入力情報（生データ）
    rankings.to_csv(output_dir / "all_rankings.csv", index=False)
    if not meta.empty:
        meta.to_csv(output_dir / "all_meta.csv", index=False)

    # --- 2. クロスモデル分析 ---
    pair_corr = cm.pairwise_rank_correlation(rankings)
    pair_corr.to_csv(cross_dir / "pairwise_rank_correlation.csv", index=False)

    tie_rates = cm.tie_rates(rankings)
    tie_rates.to_csv(cross_dir / "tie_rates.csv")

    divergent = cm.divergent_cases(rankings, top_n=30)
    divergent.to_csv(cross_dir / "divergent_teams.csv", index=False)

    team_avg = cm.team_avg_rank(rankings)
    team_avg.to_csv(output_dir / "per_team_ranks.csv", index=False)

    crit_corr = cm.criteria_correlation(rankings)
    crit_corr.to_csv(cross_dir / "criteria_correlation.csv", index=False)

    # --- 3. メタ情報相関（B5 系の監査） ---
    merged = corr_mod.join_rank_with_meta(rankings, meta)
    avg_by_player = corr_mod.avg_rank_by_player(merged)
    avg_by_player.to_csv(meta_dir / "avg_rank_by_player.csv", index=False)

    numeric_corr_per_crit = corr_mod.numeric_meta_correlation(merged)
    numeric_corr_per_crit.to_csv(
        meta_dir / "numeric_meta_correlation_per_criterion.csv", index=False
    )

    numeric_corr_avg = corr_mod.numeric_meta_correlation_avg(avg_by_player)
    numeric_corr_avg.to_csv(
        meta_dir / "numeric_meta_correlation_avg.csv", index=False
    )

    for category in ("death_cause", "won", "role"):
        cat_df = corr_mod.avg_rank_by_category(avg_by_player, category)
        if not cat_df.empty:
            cat_df.to_csv(meta_dir / f"avg_rank_by_{category}.csv", index=False)

    # --- 4. プロット ---
    if make_plots:
        try:
            plots.plot_rank_correlation_heatmap(
                pair_corr, plot_dir / "rank_correlation_heatmap.png"
            )
            plots.plot_tie_rates_bar(
                tie_rates, plot_dir / "tie_rates_bar.png"
            )
            plots.plot_meta_scatter_panel(
                avg_by_player, plot_dir / "meta_scatter_panel.png"
            )
            plots.plot_team_avg_rank(
                team_avg, plot_dir / "team_avg_per_model.png"
            )
            plots.plot_criteria_correlation(
                crit_corr, plot_dir / "criteria_correlation_heatmap.png"
            )
        except Exception as e:
            logger.exception(f"プロット生成中にエラー: {e}")

    # --- 5. Summary ---
    write_summary(
        output_dir / "summary.md",
        input_dir=input_dir,
        models=models,
        rankings=rankings,
        pair_corr=pair_corr,
        tie_rates=tie_rates,
        divergent=divergent,
        numeric_corr_avg=numeric_corr_avg,
        team_avg=team_avg,
        avg_by_player=avg_by_player,
        crit_corr=crit_corr,
    )
