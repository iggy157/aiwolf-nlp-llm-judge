"""prompts.yaml の読み込みを一箇所に集約するローダー."""

from pathlib import Path
from typing import Any

import yaml

from src.llm.client import PromptTemplates


def load_prompt_templates(config: dict[str, Any]) -> PromptTemplates:
    """設定辞書から prompts.yaml を読み込んで PromptTemplates を返す.

    旧来のテンプレートでは "developer" キーがシステムプロンプトに使われていたため
    互換性のために "developer" もしくは "system" のどちらかを許容する。

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

    return PromptTemplates(system=system_template, user=user_template)
