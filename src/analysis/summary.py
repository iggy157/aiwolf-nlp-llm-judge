"""分析結果サマリの Markdown 生成."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


def write_summary(
    output_path: Path,
    *,
    input_dir: Path,
    models: list[str],
    rankings: pd.DataFrame,
    pair_corr: pd.DataFrame,
    tie_rates: pd.DataFrame,
    divergent: pd.DataFrame,
    numeric_corr_avg: pd.DataFrame,
    team_avg: pd.DataFrame,
    avg_by_player: pd.DataFrame,
    crit_corr: pd.DataFrame,
) -> None:
    """summary.md を書き出す."""
    lines: list[str] = []
    lines.append("# LLM-Judge クロスモデル分析サマリ")
    lines.append("")
    lines.append(f"- 入力: `{input_dir}`")
    lines.append(f"- 生成日時: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- モデル数: {len(models)}（{', '.join(models)}）")
    lines.append(f"- ゲーム数: {rankings['ulid'].nunique()}")
    lines.append(f"- 評価基準数: {rankings['criterion'].nunique()}")
    lines.append("")

    # --- モデル間相関 ---
    lines.append("## モデル間 rank 相関（Spearman）")
    lines.append("")
    if pair_corr.empty:
        lines.append("（モデル数が1のためスキップ）")
    else:
        avg_r = pair_corr.groupby(["model_a", "model_b"])["spearman_r"].mean()
        lines.append("基準を跨いだ平均相関係数:")
        lines.append("")
        lines.append("| model_a | model_b | mean Spearman r |")
        lines.append("|---|---|---|")
        for (a, b), r in avg_r.items():
            lines.append(f"| {a} | {b} | {r:+.3f} |")
        lines.append("")
        lines.append("基準別の詳細は [`cross_model/pairwise_rank_correlation.csv`](cross_model/pairwise_rank_correlation.csv)")
    lines.append("")

    # --- 同順位率 ---
    lines.append("## 同順位（タイ）使用率")
    lines.append("")
    if tie_rates.empty:
        lines.append("（データなし）")
    else:
        cols = list(tie_rates.columns)
        lines.append("| model | " + " | ".join(cols) + " |")
        lines.append("|" + "---|" * (len(cols) + 1))
        for model, row in tie_rates.iterrows():
            cells = [f"{row[c]:.0%}" for c in cols]
            lines.append(f"| {model} | " + " | ".join(cells) + " |")
    lines.append("")

    # --- メタ情報相関（B5 系構造指標） ---
    lines.append("## メタ情報 × avg_rank 相関（バイアス監査）")
    lines.append("")
    lines.append(
        "プレイヤー単位の avg_rank（全基準平均）と、ゲームに依存しない正規化メタ指標の Spearman 相関。"
        "符号が正なら「指標が大きいほど rank が悪い（低評価）」を意味する。"
    )
    lines.append("")
    if numeric_corr_avg.empty:
        lines.append("（メタ情報が抽出できませんでした）")
    else:
        pivot = numeric_corr_avg.pivot_table(
            index="model", columns="meta", values="spearman_r"
        )
        cols = list(pivot.columns)
        lines.append("| model | " + " | ".join(cols) + " | n |")
        lines.append("|" + "---|" * (len(cols) + 2))
        n_by_model = numeric_corr_avg.groupby("model")["n"].max()
        for model, row in pivot.iterrows():
            cells = [f"{row[c]:+.3f}" if pd.notna(row[c]) else "-" for c in cols]
            lines.append(
                f"| {model} | " + " | ".join(cells)
                + f" | {n_by_model.get(model, '-')} |"
            )
        lines.append("")
        lines.append("**指標の意味**:")
        lines.append("- `vote_rate`: 被投票数 ÷ ゲーム内総投票数")
        lines.append("- `survival_rate`: 脱落日 ÷ ゲーム最大日数")
        lines.append("- `talk_rate`: 有効発話数 ÷ ゲーム内総有効発話数（\"Over\" 単独は除外）")
        lines.append("")
        lines.append("**注意**: rank は「低いほど良い」ため、`vote_rate` と正の相関が出ると "
                     "「票を集めたプレイヤーほど低評価」というB5的なバイアスの兆候。")
    lines.append("")

    # --- カテゴリ別 avg_rank（死因・勝敗・役職） ---
    lines.append("## カテゴリ別 avg_rank")
    lines.append("")
    for category, label in [("death_cause", "死因"), ("won", "勝敗"), ("role", "役職")]:
        if category not in avg_by_player.columns:
            continue
        cat = (
            avg_by_player.groupby(["model", category], dropna=False)["avg_rank"]
            .mean()
            .unstack(category)
        )
        if cat.empty:
            continue
        lines.append(f"### {label}別")
        lines.append("")
        cols = [str(c) for c in cat.columns]
        lines.append("| model | " + " | ".join(cols) + " |")
        lines.append("|" + "---|" * (len(cols) + 1))
        for model, row in cat.iterrows():
            cells = [f"{row[c]:.2f}" if pd.notna(row[c]) else "-" for c in cat.columns]
            lines.append(f"| {model} | " + " | ".join(cells) + " |")
        lines.append("")

    # --- 乖離大の事例 top 10 ---
    lines.append("## モデル間で評価が割れた事例（max-min 降順 top 10）")
    lines.append("")
    if divergent.empty:
        lines.append("（モデル数が1のためスキップ）")
    else:
        cols = [c for c in divergent.columns
                if c not in ("ulid", "team", "player_name", "criterion",
                             "max_min_diff", "n_models")]
        header = "| ulid | team | player | criterion | " + " | ".join(cols) + " | max-min |"
        sep = "|" + "---|" * (5 + len(cols))
        lines.append(header)
        lines.append(sep)
        for _, row in divergent.head(10).iterrows():
            cells = [f"{row[c]:.0f}" if pd.notna(row[c]) else "-" for c in cols]
            lines.append(
                f"| `{row['ulid'][:8]}` | {row['team']} | {row['player_name']} | "
                f"{row['criterion']} | " + " | ".join(cells)
                + f" | {row['max_min_diff']:.0f} |"
            )
        lines.append("")
        lines.append("詳細は [`cross_model/divergent_teams.csv`](cross_model/divergent_teams.csv)")
    lines.append("")

    # --- チーム別の総合順位 ---
    lines.append("## チーム別 avg_rank（全モデル × 全基準の平均）")
    lines.append("")
    if not team_avg.empty:
        overall = (
            team_avg.groupby("team")["avg_rank"].mean()
            .sort_values()
            .reset_index()
        )
        lines.append("| 順位 | team | avg_rank |")
        lines.append("|---|---|---|")
        for i, row in enumerate(overall.itertuples(index=False), 1):
            lines.append(f"| {i} | {row.team} | {row.avg_rank:.2f} |")
        lines.append("")

    # --- 基準間相関 ---
    lines.append("## 基準間の相関（同一モデル内）")
    lines.append("")
    if crit_corr.empty:
        lines.append("（データ不足）")
    else:
        for model, sub in crit_corr.groupby("model"):
            lines.append(f"### {model}")
            lines.append("")
            criteria = sorted(set(sub["criterion_a"]) | set(sub["criterion_b"]))
            mat = pd.DataFrame(index=criteria, columns=criteria, dtype=float)
            for _, row in sub.iterrows():
                mat.loc[row["criterion_a"], row["criterion_b"]] = row["spearman_r"]
                mat.loc[row["criterion_b"], row["criterion_a"]] = row["spearman_r"]
            for c in criteria:
                mat.loc[c, c] = 1.0
            lines.append("| | " + " | ".join(criteria) + " |")
            lines.append("|" + "---|" * (len(criteria) + 1))
            for r in criteria:
                cells = [f"{mat.loc[r, c]:+.2f}" if pd.notna(mat.loc[r, c]) else "-"
                         for c in criteria]
                lines.append(f"| {r} | " + " | ".join(cells) + " |")
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("生成物一覧:")
    lines.append("- `all_rankings.csv` / `all_meta.csv`: 入力のフラット化データ")
    lines.append("- `per_team_ranks.csv`: チーム × 基準 × モデルの avg_rank")
    lines.append("- `cross_model/`: モデル間比較系")
    lines.append("- `meta_correlation/`: メタ情報相関とカテゴリ集計")
    lines.append("- `plots/`: ヒートマップ・散布図 等")

    output_path.write_text("\n".join(lines), encoding="utf-8")
