"""prompts.yaml の読み込みを一箇所に集約するローダー."""

from pathlib import Path
from typing import Any

import yaml

from src.llm.client import PromptTemplates


# preface を組み立てる際のセクション見出し。
# A 部品（最低水準系）と B 部品（評価軸外宣言系）でセクションを分けるため、
# 部品名のプレフィックスでグループを判定する。
_BASELINE_SECTION_HEADERS = {
    "min_standard": "## 評価の最低条件（適切な発話として扱う最低ライン）",
    "out_of_scope": "## 評価軸ではないもの（いかなる順位変動の根拠ともしない）",
}


def _classify_component(name: str) -> str:
    """部品名から所属セクションを判定.

    "exclude_" で始まる部品は評価軸外宣言（B）、それ以外は最低水準系（A）。
    """
    return "out_of_scope" if name.startswith("exclude_") else "min_standard"


def _build_preface(
    focus: str,
    apply_list: list[str],
    library: dict[str, str],
) -> str:
    """focus + 適用部品リストから1つの preface 文字列を組み立てる.

    Args:
        focus: 評価軸の宣言テキスト
        apply_list: 適用する部品名のリスト（library のキー）
        library: baseline_components 辞書（部品名 → 本文）

    Returns:
        preface 文字列。focus + セクション化された部品本文。
    """
    parts: list[str] = []
    focus_text = focus.strip()
    if focus_text:
        parts.append(f"## この基準の焦点\n{focus_text}")

    # A 部品と B 部品をそれぞれの apply 順で集約
    grouped: dict[str, list[str]] = {"min_standard": [], "out_of_scope": []}
    for comp_name in apply_list:
        body = library.get(comp_name)
        if not body:
            continue
        section = _classify_component(comp_name)
        grouped[section].append(f"- {body.strip()}")

    for key in ("min_standard", "out_of_scope"):
        items = grouped[key]
        if items:
            header = _BASELINE_SECTION_HEADERS[key]
            parts.append(f"{header}\n" + "\n".join(items))

    return "\n\n".join(parts)


def load_prompt_templates(config: dict[str, Any]) -> PromptTemplates:
    """設定辞書から prompts.yaml を読み込んで PromptTemplates を返す.

    旧来のテンプレートでは "developer" キーがシステムプロンプトに使われていたため
    互換性のために "developer" もしくは "system" のどちらかを許容する。

    criterion_preface は2形式をサポート:
    - 旧形式: criterion_preface[name] が文字列（そのまま preface として使う）
    - 新形式: criterion_preface[name] が {focus: str, apply: list[str]} の辞書。
      baseline_components ライブラリから apply 配列で指定された部品を組み立てて preface を生成する。

    Args:
        config: アプリケーション設定辞書（config["llm"]["prompt_yml"] を参照）

    Returns:
        ロード済みの PromptTemplates

    Raises:
        KeyError: 必須キーが設定にない場合
        FileNotFoundError: prompts.yaml が見つからない場合
        ValueError: prompts.yaml の内容が不正な場合
    """
    try:
        prompt_yml_path = Path(config["llm"]["prompt_yml"])
    except KeyError as e:
        raise KeyError(f"必要な設定キーが見つかりません: {e}") from e

    if not prompt_yml_path.is_file():
        raise FileNotFoundError(
            f"プロンプトYAMLファイルが見つかりません: {prompt_yml_path}"
        )

    try:
        with prompt_yml_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"プロンプトYAMLファイルの解析に失敗しました: {e}") from e

    system_template = data.get("developer") or data.get("system")
    user_template = data.get("user")
    if not system_template or not user_template:
        raise ValueError(
            "prompts.yaml には 'developer'（または 'system'）と 'user' の両方が必要です"
        )

    raw_prefaces = data.get("criterion_preface") or {}
    if not isinstance(raw_prefaces, dict):
        raise ValueError(
            "prompts.yaml の 'criterion_preface' は基準名→preface定義の辞書である必要があります"
        )

    library = data.get("baseline_components") or {}
    if not isinstance(library, dict):
        raise ValueError(
            "prompts.yaml の 'baseline_components' は部品名→本文の辞書である必要があります"
        )
    library = {str(name): str(body) for name, body in library.items() if body}

    criterion_prefaces: dict[str, str] = {}
    for name, spec in raw_prefaces.items():
        if not spec:
            continue
        # 後方互換: 旧来は文字列を直接 preface としていた
        if isinstance(spec, str):
            criterion_prefaces[str(name)] = spec.strip()
            continue
        if not isinstance(spec, dict):
            raise ValueError(
                f"criterion_preface[{name}] は文字列または "
                "{focus, apply} 辞書である必要があります"
            )
        focus = str(spec.get("focus", "")).strip()
        apply_raw = spec.get("apply") or []
        if not isinstance(apply_raw, list):
            raise ValueError(
                f"criterion_preface[{name}].apply はリストである必要があります"
            )
        apply_list = [str(x) for x in apply_raw]
        # 未知の部品名は警告レベルで無視するか厳格に弾くか。
        # 設定ミスの早期発見のため、ここでは未定義部品は明示的にエラーにする。
        unknown = [n for n in apply_list if n not in library]
        if unknown:
            raise ValueError(
                f"criterion_preface[{name}].apply に未定義の部品があります: {unknown}"
            )

        built = _build_preface(focus, apply_list, library)
        if built:
            criterion_prefaces[str(name)] = built

    return PromptTemplates(
        system=system_template,
        user=user_template,
        criterion_prefaces=criterion_prefaces,
    )
