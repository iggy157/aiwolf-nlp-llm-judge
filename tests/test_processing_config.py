"""ProcessingConfig 読み込みと filter_models のテスト."""

import pytest

from src.processor.models import ProcessingConfig
from src.processor.models.exceptions import ConfigurationError


def _base_config() -> dict:
    return {
        "path": {"env": "config/.env", "evaluation_criteria": "config/x.yaml"},
        "llm": {
            "prompt_yml": "config/prompts.yaml",
            "models": [
                {"id": "gpt", "provider": "openai", "model": "gpt-4o", "api_key_env": "K"},
                {
                    "id": "claude",
                    "provider": "anthropic",
                    "model": "claude-x",
                    "api_key_env": "K2",
                },
            ],
        },
        "game": {"format": "main_match"},
        "processing": {
            "input_dir": "data/input",
            "output_dir": "data/output",
            "max_workers": 2,
        },
        "settings_path": "config/settings.yaml",
    }


class TestProcessingConfig:
    def test_defaults_applied(self) -> None:
        pc = ProcessingConfig.from_config_dict(_base_config())
        assert pc.parallel_models is False
        assert pc.dry_run is True
        assert pc.dry_run_strict is False
        assert pc.use_batch_api is False
        assert pc.enable_caching is True
        assert pc.max_retries == 3
        assert pc.evaluation_workers == 8

    def test_loads_models_in_order(self) -> None:
        pc = ProcessingConfig.from_config_dict(_base_config())
        assert [m.id for m in pc.models] == ["gpt", "claude"]

    def test_empty_models_raises(self) -> None:
        config = _base_config()
        config["llm"]["models"] = []
        with pytest.raises(ConfigurationError, match="llm.models is required"):
            ProcessingConfig.from_config_dict(config)

    def test_duplicate_model_id_raises(self) -> None:
        config = _base_config()
        config["llm"]["models"].append(
            {"id": "gpt", "provider": "openai", "model": "gpt-4o", "api_key_env": "K"}
        )
        with pytest.raises(ConfigurationError, match="Duplicate model id"):
            ProcessingConfig.from_config_dict(config)

    def test_unknown_provider_raises(self) -> None:
        config = _base_config()
        config["llm"]["models"] = [
            {"id": "x", "provider": "bogus", "model": "y"}
        ]
        with pytest.raises(ConfigurationError, match="Unknown LLM provider"):
            ProcessingConfig.from_config_dict(config)

    def test_filter_models_subset(self) -> None:
        pc = ProcessingConfig.from_config_dict(_base_config())
        filtered = pc.filter_models(["claude"])
        assert [m.id for m in filtered.models] == ["claude"]

    def test_filter_models_no_filter_returns_self(self) -> None:
        pc = ProcessingConfig.from_config_dict(_base_config())
        assert pc.filter_models(None).models == pc.models
        assert pc.filter_models([]).models == pc.models

    def test_filter_models_unknown_id_raises(self) -> None:
        pc = ProcessingConfig.from_config_dict(_base_config())
        with pytest.raises(ConfigurationError, match="Unknown model ids"):
            pc.filter_models(["nonexistent"])
