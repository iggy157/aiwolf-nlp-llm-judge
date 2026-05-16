"""BatchOrchestrator のヘルパメソッドのテスト."""

from unittest.mock import MagicMock

from src.processor.batch_orchestrator import CUSTOM_ID_SEPARATOR, BatchOrchestrator


class TestCustomIdRoundtrip:
    def test_make_and_parse(self) -> None:
        cid = BatchOrchestrator._make_custom_id("game1", "natural_expression")
        assert cid == f"game1{CUSTOM_ID_SEPARATOR}natural_expression"
        game_id, criteria_name = BatchOrchestrator._parse_custom_id(cid)
        assert game_id == "game1"
        assert criteria_name == "natural_expression"

    def test_parse_missing_separator(self) -> None:
        gid, cname = BatchOrchestrator._parse_custom_id("broken")
        assert gid == "broken"
        assert cname == ""

    def test_separator_in_game_id_only_splits_first(self) -> None:
        """criteria_name 内に separator が含まれていても、game_id 側は最初の出現で区切られる."""
        # 仕様: split(sep, 1) なので criteria に separator が混じる可能性のみ
        cid = f"g1{CUSTOM_ID_SEPARATOR}weird{CUSTOM_ID_SEPARATOR}name"
        gid, cname = BatchOrchestrator._parse_custom_id(cid)
        assert gid == "g1"
        assert cname == f"weird{CUSTOM_ID_SEPARATOR}name"


class TestBuildBatchRequests:
    def test_one_request_per_game_criteria_pair(self) -> None:
        """N ゲーム × M 基準 → N×M 件のリクエストになる."""
        orch = BatchOrchestrator.__new__(BatchOrchestrator)
        orch.model_config = MagicMock(id="m")

        # contexts: 2 ゲーム × 2 基準
        criteria_a = MagicMock(description="d-a")
        criteria_a.name = "natural"
        criteria_b = MagicMock(description="d-b")
        criteria_b.name = "team_play"

        from src.processor.pipeline.game_context import GameContext

        ctx1 = GameContext(
            game_log=MagicMock(game_id="g1"),
            game_info=MagicMock(),
            criteria=[criteria_a, criteria_b],
            formatted_data=[{"day": 0}],
            character_info="info1",
            agent_to_team_mapping={},
        )
        ctx2 = GameContext(
            game_log=MagicMock(game_id="g2"),
            game_info=MagicMock(),
            criteria=[criteria_a],
            formatted_data=[{"day": 0}],
            character_info="info2",
            agent_to_team_mapping={},
        )

        requests = orch._build_batch_requests([ctx1, ctx2])
        # g1×2 + g2×1 = 3 件
        assert len(requests) == 3
        custom_ids = [r.custom_id for r in requests]
        assert f"g1{CUSTOM_ID_SEPARATOR}natural" in custom_ids
        assert f"g1{CUSTOM_ID_SEPARATOR}team_play" in custom_ids
        assert f"g2{CUSTOM_ID_SEPARATOR}natural" in custom_ids
