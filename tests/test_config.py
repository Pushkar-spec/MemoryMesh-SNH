"""Tests for MemoryMesh configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from memorymesh.config import (
    AttributeConfig,
    Cardinality,
    ConfigFileNotFoundError,
    ConfigParseError,
    ConfigValidationError,
    EngineConfig,
    FuzzyConfig,
    LLMConfig,
    LLMProvider,
    SourceConfig,
    SourceType,
    ThresholdsConfig,
    ValueType,
    load_app_config,
    load_attributes_config,
    load_engine_config,
    load_entities_config,
    load_sources_config,
    load_yaml_file,
)


def test_load_all_configurations() -> None:
    """Verify that all four documented YAML files load and validate properly."""
    app_cfg = load_app_config()

    # Engine config structure
    assert isinstance(app_cfg.engine, EngineConfig)
    assert isinstance(app_cfg.engine.thresholds, ThresholdsConfig)
    assert isinstance(app_cfg.engine.fuzzy, FuzzyConfig)
    assert isinstance(app_cfg.engine.llm, LLMConfig)

    # Sources
    assert len(app_cfg.sources) == 5
    source_ids = {s.id for s in app_cfg.sources}
    assert source_ids == {"sysadmin_priya", "ci_monitor", "wiki", "dev_chat", "intern_chat"}

    # Attributes
    assert len(app_cfg.attributes) == 7
    attr_names = {a.attribute for a in app_cfg.attributes}
    assert attr_names == {
        "python_version",
        "os_release",
        "owner_team",
        "deploy_day",
        "service_status",
        "runs_service",
        "compatibility_note",
    }

    # Entities
    assert len(app_cfg.entities) == 2
    entity_names = {e.name for e in app_cfg.entities}
    assert entity_names == {"Server A", "Library X"}


def test_engine_threshold_values() -> None:
    """Verify documented numeric threshold and config values match the specification."""
    cfg = load_engine_config()
    t = cfg.thresholds

    assert t.min_extract_confidence == 0.40
    assert t.replace_margin == 0.80
    assert t.reject_ratio == 0.50
    assert t.revert_margin == 1.00
    assert t.max_confidence == 0.99
    assert t.review_penalty == 0.50
    assert t.stale_floor == 0.30
    assert t.dispute_autoresolve_ratio == 0.50

    f = cfg.fuzzy
    assert f.entity_auto_match == 90
    assert f.entity_tiebreak_min == 75
    assert f.text_duplicate_candidate == 70

    a = cfg.ablation
    assert a.use_temporal is True
    assert a.use_trust is True
    assert a.use_cardinality is True
    assert a.use_context is True
    assert a.use_propagation is True

    llm = cfg.llm
    assert llm.concurrency == 3
    assert llm.cache is True


def test_source_reliabilities_match_specification() -> None:
    """Verify that source types and reliabilities match the specification table."""
    sources = load_sources_config()
    source_map = {s.id: s for s in sources}

    assert source_map["sysadmin_priya"].reliability == 0.90
    assert source_map["sysadmin_priya"].type == SourceType.ADMIN

    assert source_map["ci_monitor"].reliability == 0.85
    assert source_map["ci_monitor"].type == SourceType.TOOL

    assert source_map["wiki"].reliability == 0.75
    assert source_map["wiki"].type == SourceType.WIKI

    assert source_map["dev_chat"].reliability == 0.50
    assert source_map["dev_chat"].type == SourceType.CHAT

    assert source_map["intern_chat"].reliability == 0.30
    assert source_map["intern_chat"].type == SourceType.CHAT


def test_attribute_definitions() -> None:
    """Verify attribute catalog types, cardinality, and optional fields."""
    attrs = load_attributes_config()
    attr_map = {a.attribute: a for a in attrs}

    # Single-valued version
    py = attr_map["python_version"]
    assert py.cardinality == Cardinality.SINGLE
    assert py.value_type == ValueType.VERSION
    assert "python" in py.aliases

    # Multi-valued text
    runs = attr_map["runs_service"]
    assert runs.cardinality == Cardinality.MULTI
    assert runs.value_type == ValueType.TEXT

    # Enum with allowed_values and half_life_days
    status = attr_map["service_status"]
    assert status.cardinality == Cardinality.SINGLE
    assert status.value_type == ValueType.ENUM
    assert status.half_life_days == 14.0
    assert "up" in status.allowed_values
    assert "down" in status.allowed_values


def test_entity_definitions() -> None:
    """Verify curated entities and their alias mappings."""
    entities = load_entities_config()
    entity_map = {e.name: e for e in entities}

    server_a = entity_map["Server A"]
    assert server_a.type == "server"
    assert "srv-a" in server_a.aliases
    assert "server-a" in server_a.aliases

    lib_x = entity_map["Library X"]
    assert lib_x.type == "library"
    assert "libx" in lib_x.aliases


def test_app_config_helpers() -> None:
    """Verify helper lookup methods on AppConfig."""
    app_cfg = load_app_config()

    # Source lookup
    assert app_cfg.get_source("sysadmin_priya") is not None
    assert app_cfg.get_source("unknown_source") is None

    # Attribute lookup
    assert app_cfg.get_attribute("python_version") is not None
    assert app_cfg.get_attribute("nonexistent_attribute") is None

    # Entity lookup by exact name and alias (case-insensitive)
    assert app_cfg.get_entity("Server A") is not None
    assert app_cfg.get_entity("srv-a") is not None
    assert app_cfg.get_entity("SRV-A") is not None
    assert app_cfg.get_entity("unknown_server") is None


def test_source_reliability_validation() -> None:
    """Verify that source reliability outside [0.0, 1.0] is rejected."""
    with pytest.raises(ValidationError):
        SourceConfig(
            id="test",
            name="Test",
            type=SourceType.ADMIN,
            reliability=-0.05,
        )

    with pytest.raises(ValidationError):
        SourceConfig(
            id="test",
            name="Test",
            type=SourceType.ADMIN,
            reliability=1.05,
        )


def test_invalid_llm_provider_rejected() -> None:
    """Verify that an undocumented LLM provider is rejected."""
    with pytest.raises(ValidationError):
        LLMConfig(
            provider="openai",  # type: ignore[arg-type]
            fast_model="fast",
            answer_model="answer",
            concurrency=2,
        )


def test_invalid_cardinality_rejected() -> None:
    """Verify that an invalid cardinality string is rejected."""
    with pytest.raises(ValidationError):
        AttributeConfig(
            attribute="test",
            description="test attr",
            cardinality="many",  # type: ignore[arg-type]
            value_type=ValueType.TEXT,
        )


def test_invalid_value_type_rejected() -> None:
    """Verify that an unsupported value_type string is rejected."""
    with pytest.raises(ValidationError):
        AttributeConfig(
            attribute="test",
            description="test attr",
            cardinality=Cardinality.SINGLE,
            value_type="boolean",  # type: ignore[arg-type]
        )


def test_missing_required_fields_rejected(tmp_path: Path) -> None:
    """Verify that missing required configuration fields produce a validation error."""
    incomplete_engine = tmp_path / "engine.yaml"
    # Missing 'llm' and 'fuzzy' sections
    incomplete_engine.write_text("thresholds: {}\n", encoding="utf-8")

    with pytest.raises(ConfigValidationError):
        load_engine_config(tmp_path)


def test_missing_yaml_file_error(tmp_path: Path) -> None:
    """Verify that a missing configuration file raises ConfigFileNotFoundError."""
    empty_dir = tmp_path / "empty_config"
    empty_dir.mkdir()

    with pytest.raises(ConfigFileNotFoundError):
        load_engine_config(empty_dir)

    with pytest.raises(ConfigFileNotFoundError):
        load_app_config(empty_dir)


def test_malformed_yaml_error(tmp_path: Path) -> None:
    """Verify that malformed YAML syntax raises ConfigParseError."""
    bad_yaml = tmp_path / "engine.yaml"
    bad_yaml.write_text("thresholds: [unclosed list", encoding="utf-8")

    with pytest.raises(ConfigParseError):
        load_engine_config(tmp_path)

    with pytest.raises(ConfigParseError):
        load_yaml_file(bad_yaml)


def test_yaml_non_dict_error(tmp_path: Path) -> None:
    """Verify that a YAML file containing a list instead of a mapping raises ConfigParseError."""
    list_yaml = tmp_path / "engine.yaml"
    list_yaml.write_text("- item1\n- item2\n", encoding="utf-8")

    with pytest.raises(ConfigParseError):
        load_engine_config(tmp_path)


def test_llm_provider_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that the LLM_PROVIDER environment variable overrides engine.yaml setting."""
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    cfg = load_engine_config()
    assert cfg.llm.provider == LLMProvider.FAKE

    monkeypatch.setenv("LLM_PROVIDER", "invalid_provider")
    with pytest.raises(ConfigValidationError):
        load_engine_config()


def test_confidence_threshold_configuration_validation() -> None:
    """Verify that invalid confidence parameters in ThresholdsConfig are rejected."""
    # Valid baseline parameters
    base_kwargs = {
        "min_extract_confidence": 0.40,
        "replace_margin": 0.80,
        "reject_ratio": 0.50,
        "revert_margin": 1.00,
        "max_confidence": 0.99,
        "review_penalty": 0.50,
        "stale_floor": 0.30,
        "dispute_autoresolve_ratio": 0.50,
    }

    # max_confidence must be strictly positive and <= 1.0
    with pytest.raises(ValidationError):
        ThresholdsConfig(**{**base_kwargs, "max_confidence": 0.0})
    with pytest.raises(ValidationError):
        ThresholdsConfig(**{**base_kwargs, "max_confidence": -0.1})
    with pytest.raises(ValidationError):
        ThresholdsConfig(**{**base_kwargs, "max_confidence": 1.1})

    # review_penalty in [0.0, 1.0]
    with pytest.raises(ValidationError):
        ThresholdsConfig(**{**base_kwargs, "review_penalty": -0.1})
    with pytest.raises(ValidationError):
        ThresholdsConfig(**{**base_kwargs, "review_penalty": 1.1})

    # stale_floor in [0.0, 1.0]
    with pytest.raises(ValidationError):
        ThresholdsConfig(**{**base_kwargs, "stale_floor": -0.1})
    with pytest.raises(ValidationError):
        ThresholdsConfig(**{**base_kwargs, "stale_floor": 1.1})


def test_attribute_half_life_validation() -> None:
    """Verify that non-positive half_life_days values are rejected in AttributeConfig."""
    with pytest.raises(ValidationError):
        AttributeConfig(
            attribute="test",
            description="test",
            cardinality=Cardinality.SINGLE,
            value_type=ValueType.TEXT,
            half_life_days=0.0,
        )

    with pytest.raises(ValidationError):
        AttributeConfig(
            attribute="test",
            description="test",
            cardinality=Cardinality.SINGLE,
            value_type=ValueType.TEXT,
            half_life_days=-14.0,
        )
