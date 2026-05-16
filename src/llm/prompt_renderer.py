"""PromptTemplates の Jinja2 レンダリングを集約するユーティリティ.

各プロバイダクライアント（同期 + バッチ）で同じレンダリングを行うため、
ここに集約してロジック重複を防ぐ。
"""

from jinja2 import Template

from src.llm.client import PromptTemplates


# Anthropic のキャッシュマーカー用に user prompt を分割する際のヘッダ文字列。
# anthropic_client / anthropic_batch から共通利用する。
EVALUATION_BLOCK_HEADER = "## 評価基準"
LOG_BLOCK_HEADER = "## 評価対象のログ"


def render_system(templates: PromptTemplates) -> str:
    """system プロンプトテンプレートを文字列に展開."""
    return Template(templates.system).render().strip()


def render_user(
    templates: PromptTemplates,
    character_info: str,
    criteria_description: str,
    log_json: str,
) -> str:
    """user プロンプトテンプレートを変数差し込みで展開."""
    return (
        Template(templates.user)
        .render(
            character_info=character_info,
            criteria_description=criteria_description,
            log=log_json,
        )
        .strip()
    )


def split_user_prompt(
    templates: PromptTemplates,
    character_info: str,
    criteria_description: str,
    log_json: str,
) -> tuple[str, str]:
    """user プロンプトをキャッシュ対象 prefix と可変部に分割.

    Anthropic の cache_control マーカー、および Gemini の cached_content 利用時に
    「prefix（system 説明 + character_info）」と「変動部（criteria + log）」を
    分けて扱うのに使う。

    prompts.yaml の user テンプレートは「説明文 + character_info + criteria + log」の
    順を想定。criteria_description と log を空文字でレンダリングすることで prefix を取り出す。
    """
    prefix_text = (
        Template(templates.user)
        .render(
            character_info=character_info,
            criteria_description="",
            log="",
        )
        .rstrip()
    )
    varying_text = (
        f"{EVALUATION_BLOCK_HEADER}\n\n{criteria_description}\n\n"
        f"{LOG_BLOCK_HEADER}\n\n{log_json}"
    )
    return prefix_text, varying_text
