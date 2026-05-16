from dataclasses import dataclass
from enum import Enum


class GameFormat(Enum):
    """ゲーム形式を表す列挙型."""

    SELF_MATCH = "self_match"
    MAIN_MATCH = "main_match"


@dataclass
class GameInfo:
    """ゲーム情報を表すデータクラス."""

    game_format: GameFormat
    player_count: int
    werewolf_count: int
    game_id: str = ""


@dataclass
class PlayerInfo:
    """ゲーム参加者を表すデータクラス."""

    index: int
    full_team_name: str
    team: str


@dataclass
class CharacterInfo:
    """キャラクター情報を表すデータクラス."""

    name: str
    profile: str
