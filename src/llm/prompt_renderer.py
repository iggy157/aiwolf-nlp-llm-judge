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


def _resolve_preface(
    templates: PromptTemplates, criterion_name: str | None
) -> str:
    """基準名から user 先頭に挿入する preface 文字列を取り出す.

    未指定 / 未定義の基準は空文字を返す（後方互換）。
    """
    if not criterion_name:
        return ""
    return templates.criterion_prefaces.get(criterion_name, "").strip()


def render_user(
    templates: PromptTemplates,
    character_info: str,
    criteria_description: str,
    log_json: str,
    criterion_name: str | None = None,
) -> str:
    """user プロンプトテンプレートを変数差し込みで展開."""
    return (
        Template(templates.user)
        .render(
            criterion_preface=_resolve_preface(templates, criterion_name),
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
    criterion_name: str | None = None,
) -> tuple[str, str]:
    """user プロンプトをキャッシュ対象 prefix と可変部に分割.

    Anthropic の cache_control マーカー、および Gemini の cached_content 利用時に
    「prefix（基準別 preface + 説明文 + character_info）」と「変動部（criteria + log）」を
    分けて扱うのに使う。

    基準別 preface は基準ごとに内容が変わるが、同一基準を複数ゲームで使う場合は
    再利用される。同一ゲームの複数基準実行ではキャッシュ衝突が起きないよう、
    本関数の呼び出し側は基準ごとに別ハンドルとして管理することを想定する。

    prompts.yaml の user テンプレートは「preface + 説明文 + character_info + criteria + log」の
    順を想定。criteria_description と log を空文字でレンダリングすることで prefix を取り出す。
    """
    prefix_text = (
        Template(templates.user)
        .render(
            criterion_preface=_resolve_preface(templates, criterion_name),
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
