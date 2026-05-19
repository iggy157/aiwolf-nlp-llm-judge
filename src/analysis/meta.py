"""ゲームログからメタ情報（被投票率・生存率・発話率・役職・勝敗・死因）を抽出.

すべての量はゲーム長・人数に依存しないよう正規化（比率 or カテゴリ）して返す。
5人人狼でも9人人狼でも同じ分析パイプラインが回るように。
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from src.aiwolf_log.game_log import AIWolfGameLog
from src.llm.formatter import GameLogFormatter

logger = logging.getLogger(__name__)


def extract_meta(
    input_data_dir: Path,
    ulid_to_logfile: dict[str, str],
    ulids: list[str],
) -> pd.DataFrame:
    """指定 ULID 群のメタ情報を抽出し、(ulid, team) 単位の DataFrame で返す.

    Returns:
        DataFrame with columns:
          ulid, team, player_name, role, vote_rate, survival_rate,
          talk_rate, death_cause, won

        - vote_rate     : 被投票数 / ゲーム内総投票数 (0〜1)
        - survival_rate : (脱落日数 or 最終日) / ゲーム最大日数 (0〜1)
        - talk_rate     : 自身の有効発話数 / ゲーム内総有効発話数 (0〜1)
        - death_cause   : "survive" | "execute" | "attack"
        - won           : True | False（自陣営が勝利したか）
    """
    rows: list[dict[str, Any]] = []
    for ulid in ulids:
        fn = ulid_to_logfile.get(ulid)
        if not fn:
            logger.warning(f"ULID -> ログファイル未解決: {ulid}")
            continue
        try:
            per_game = _extract_one_game(input_data_dir, fn)
        except Exception as e:
            logger.warning(f"メタ抽出失敗 ulid={ulid} fn={fn}: {e}")
            continue
        for row in per_game:
            row["ulid"] = ulid
            rows.append(row)
    if not rows:
        logger.warning("メタ情報が空")
        return pd.DataFrame(
            columns=["ulid", "team", "player_name", "role", "vote_rate",
                     "survival_rate", "talk_rate", "death_cause", "won"]
        )
    return pd.DataFrame(rows)


def _extract_one_game(input_data_dir: Path, file_name: str) -> list[dict[str, Any]]:
    """1ゲームのメタ情報を抽出.

    Returns:
        プレイヤー単位の dict のリスト
    """
    gl = AIWolfGameLog(input_dir=input_data_dir, file_name=file_name)
    fmt = GameLogFormatter(gl, {"processing": {"encoding": "utf-8"}})
    # result も含めて取得（メタ抽出のため）
    data = fmt.convert_to_jsonl()

    # player_name -> {team, role}
    player_meta: dict[str, dict[str, str]] = {}
    for e in data:
        if e.get("action") == "status":
            p = e.get("player_name")
            if not p:
                continue
            player_meta.setdefault(p, {})["team"] = e.get("team_name", "")
            player_meta[p]["role"] = e.get("role", "")
            # 生存状態は status エントリで日次更新される

    if not player_meta:
        return []

    # 1日目以降の day 値の最大（ゲーム最大日数）
    max_day = max((e.get("day", 0) for e in data), default=0)
    if max_day <= 0:
        max_day = 1  # ゼロ除算回避

    # 各 player の脱落日: status の alive_status=DEAD が初めて出る day
    death_day: dict[str, int] = {}
    death_cause: dict[str, str] = {}  # survive / execute / attack
    for e in data:
        if e.get("action") == "status":
            p = e.get("player_name")
            if p and e.get("alive_status") == "DEAD" and p not in death_day:
                death_day[p] = e.get("day", max_day)
    # 死因（execute or attack）を後で上書き
    for e in data:
        if e.get("action") == "execute":
            p = e.get("executed_player") or e.get("executed_player_index")
            if p:
                death_cause[p] = "execute"
        elif e.get("action") == "attack":
            p = e.get("target") or e.get("target_index")  # attack はターゲット
            if p:
                death_cause[p] = "attack"

    # 被投票数（player_name -> count）
    vote_counts: Counter[str] = Counter()
    total_votes = 0
    for e in data:
        if e.get("action") == "vote":
            t = e.get("target")
            if t:
                vote_counts[t] += 1
                total_votes += 1

    # 有効発話数（"Over" のみは除外）と総有効発話数
    talk_counts: Counter[str] = Counter()
    total_talks = 0
    for e in data:
        if e.get("action") == "talk":
            text = (e.get("text") or "").strip()
            if text and text.lower() != "over":
                p = e.get("speaker")
                if p:
                    talk_counts[p] += 1
                    total_talks += 1

    # 勝利チーム（result アクションがあれば）
    winning_team: str | None = None
    for e in data:
        if e.get("action") == "result":
            winning_team = e.get("winning_team") or None
            break

    # 役職→陣営マップ（人狼は WEREWOLF/POSSESSED が狼陣営）
    def faction(role: str) -> str:
        role_u = (role or "").upper()
        if role_u in ("WEREWOLF", "POSSESSED"):
            return "werewolf"
        return "villager"

    # 「勝利チーム」表記が役職名 (WEREWOLF / VILLAGER) 系か、陣営名系かは試合構成依存。
    # ここではざっくり、winning_team が WEREWOLF or 人狼/狼 を含めば狼陣営勝利と判定。
    won_faction: str | None = None
    if winning_team:
        wt = str(winning_team).upper()
        if "WEREWOLF" in wt or "WOLF" in wt or "人狼" in wt or "狼" in winning_team:
            won_faction = "werewolf"
        else:
            won_faction = "villager"

    rows: list[dict[str, Any]] = []
    for player, meta in player_meta.items():
        role = meta.get("role", "")
        team = meta.get("team", "")
        vr = (vote_counts.get(player, 0) / total_votes) if total_votes else 0.0
        tr = (talk_counts.get(player, 0) / total_talks) if total_talks else 0.0
        # 生存日数: 死亡してない → max_day、 死亡日 d → d
        d_day = death_day.get(player, max_day)
        survival_rate = d_day / max_day
        # 死因
        if player in death_cause:
            dc = death_cause[player]
        elif player in death_day:
            # status は DEAD だが execute/attack の記録がない → unknown 扱いで survive 寄せず attack 推定はしない
            dc = "unknown"
        else:
            dc = "survive"
        # 勝敗
        if won_faction is None:
            won = None
        else:
            won = faction(role) == won_faction
        rows.append({
            "team": team,
            "player_name": player,
            "role": role,
            "vote_rate": vr,
            "survival_rate": survival_rate,
            "talk_rate": tr,
            "death_cause": dc,
            "won": won,
        })
    return rows
