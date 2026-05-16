"""PromptRenderer のテスト."""

from src.llm.client import PromptTemplates
from src.llm.prompt_renderer import (
    EVALUATION_BLOCK_HEADER,
    LOG_BLOCK_HEADER,
    render_system,
    render_user,
    split_user_prompt,
)


SYSTEM_TPL = "system instruction text"
USER_TPL = (
    "## キャラクター\n\n{{ character_info }}\n\n"
    "## 評価基準\n\n{{ criteria_description }}\n\n"
    "## 評価対象のログ\n\n{{ log }}"
)


def _templates() -> PromptTemplates:
    return PromptTemplates(system=SYSTEM_TPL, user=USER_TPL)


class TestPromptRenderer:
    def test_render_system_strips_whitespace(self) -> None:
        templates = PromptTemplates(system="  hello  ", user=USER_TPL)
        assert render_system(templates) == "hello"

    def test_render_user_substitutes_all_variables(self) -> None:
        rendered = render_user(
            _templates(),
            character_info="- alice: nice",
            criteria_description="natural",
            log_json='[{"day":0}]',
        )
        assert "- alice: nice" in rendered
        assert "natural" in rendered
        assert '[{"day":0}]' in rendered

    def test_split_user_prompt_prefix_is_stable(self) -> None:
        prefix, varying = split_user_prompt(
            _templates(),
            character_info="- alice: nice",
            criteria_description="natural",
            log_json='[{"day":0}]',
        )
        # prefix は character_info を含む（cacheable）
        assert "- alice: nice" in prefix
        # criteria_description / log は varying に入っている
        assert "natural" in varying
        assert '[{"day":0}]' in varying
        # varying は固定のヘッダで始まる（プロバイダ間で同形式）
        assert varying.startswith(EVALUATION_BLOCK_HEADER)
        assert LOG_BLOCK_HEADER in varying

    def test_split_user_prompt_prefix_invariant_across_criteria(self) -> None:
        """同一ゲーム内で character_info が同じなら prefix は不変."""
        prefix1, _ = split_user_prompt(
            _templates(), "- alice: nice", "criteriaA", "log1"
        )
        prefix2, _ = split_user_prompt(
            _templates(), "- alice: nice", "criteriaB", "log2"
        )
        assert prefix1 == prefix2
