"""Configuration loading and validation for MemoryMesh."""

from __future__ import annotations

import os
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError

# ==============================================================================
# Enums
# ==============================================================================


class Cardinality(StrEnum):
    """Attribute cardinality options."""

    SINGLE = "single"
    MULTI = "multi"


class ValueType(StrEnum):
    """Supported attribute value types."""

    VERSION = "version"
    ENUM = "enum"
    TEXT = "text"
    NUMBER = "number"
    DATE = "date"


class SourceType(StrEnum):
    """Categorization of information sources."""

    ADMIN = "admin"
    WIKI = "wiki"
    TOOL = "tool"
    CHAT = "chat"
    USER = "user"
    AGENT = "agent"


class LLMProvider(StrEnum):
    """Documented LLM providers."""

    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    FAKE = "fake"


# ==============================================================================
# Configuration Exceptions
# ==============================================================================


class ConfigError(Exception):
    """Base exception for all configuration errors."""


class ConfigFileNotFoundError(ConfigError, FileNotFoundError):
    """Raised when a required configuration file cannot be found."""


class ConfigParseError(ConfigError):
    """Raised when a YAML configuration file is malformed."""


class ConfigValidationError(ConfigError):
    """Raised when configuration values fail Pydantic schema validation."""


# ==============================================================================
# Engine Configuration Models
# ==============================================================================


class ThresholdsConfig(BaseModel):
    """Resolution and confidence threshold parameters."""

    min_extract_confidence: float = Field(ge=0.0, le=1.0)
    replace_margin: float = Field(ge=0.0)
    reject_ratio: float = Field(ge=0.0, le=1.0)
    revert_margin: float = Field(ge=0.0)
    max_confidence: float = Field(ge=0.0, le=1.0)
    review_penalty: float = Field(ge=0.0, le=1.0)
    stale_floor: float = Field(ge=0.0, le=1.0)
    dispute_autoresolve_ratio: float = Field(ge=0.0, le=1.0)


class FuzzyConfig(BaseModel):
    """Fuzzy matching ratio thresholds."""

    entity_auto_match: int = Field(ge=0, le=100)
    entity_tiebreak_min: int = Field(ge=0, le=100)
    text_duplicate_candidate: int = Field(ge=0, le=100)


class AblationConfig(BaseModel):
    """Feature ablation flags (all true in production)."""

    use_temporal: bool = True
    use_trust: bool = True
    use_cardinality: bool = True
    use_context: bool = True
    use_propagation: bool = True


class LLMConfig(BaseModel):
    """LLM provider and execution settings."""

    provider: LLMProvider
    fast_model: str
    answer_model: str
    concurrency: int = Field(gt=0)
    cache: bool = True


class EngineConfig(BaseModel):
    """Complete engine configuration (engine.yaml)."""

    thresholds: ThresholdsConfig
    fuzzy: FuzzyConfig
    ablation: AblationConfig = Field(default_factory=AblationConfig)
    llm: LLMConfig


# ==============================================================================
# Sources, Attributes, and Entities Configuration Models
# ==============================================================================


class SourceConfig(BaseModel):
    """Source definition and reliability rating."""

    id: str
    name: str
    type: SourceType
    reliability: float = Field(ge=0.0, le=1.0)
    notes: str = ""


class SourcesConfigFile(BaseModel):
    """Top-level container for sources.yaml."""

    sources: list[SourceConfig]


class AttributeConfig(BaseModel):
    """Attribute catalog definition."""

    attribute: str
    description: str
    cardinality: Cardinality
    value_type: ValueType
    half_life_days: float | None = Field(default=None, gt=0)
    allowed_values: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)


class AttributesConfigFile(BaseModel):
    """Top-level container for attributes.yaml."""

    attributes: list[AttributeConfig]


class EntityConfig(BaseModel):
    """Pre-curated entity alias mapping."""

    name: str
    type: str
    aliases: list[str] = Field(default_factory=list)


class EntitiesConfigFile(BaseModel):
    """Top-level container for entities.yaml."""

    entities: list[EntityConfig]


# ==============================================================================
# Application Combined Configuration
# ==============================================================================


class AppConfig(BaseModel):
    """Complete application configuration object."""

    engine: EngineConfig
    sources: list[SourceConfig]
    attributes: list[AttributeConfig]
    entities: list[EntityConfig]

    def get_source(self, source_id: str) -> SourceConfig | None:
        """Find a source definition by id."""
        for s in self.sources:
            if s.id == source_id:
                return s
        return None

    def get_attribute(self, attribute_name: str) -> AttributeConfig | None:
        """Find an attribute definition by name."""
        for a in self.attributes:
            if a.attribute == attribute_name:
                return a
        return None

    def get_entity(self, name_or_alias: str) -> EntityConfig | None:
        """Find an entity definition matching either exact name or an alias."""
        needle = name_or_alias.strip().lower()
        for e in self.entities:
            if e.name.lower() == needle:
                return e
            if any(alias.lower() == needle for alias in e.aliases):
                return e
        return None


# ==============================================================================
# Loaders
# ==============================================================================


def get_default_config_dir() -> Path:
    """Resolve the default directory containing configuration files."""
    env_dir = os.environ.get("MEMORYMESH_CONFIG_DIR")
    if env_dir:
        candidate = Path(env_dir).resolve()
        if candidate.is_dir():
            return candidate

    cwd_candidate = Path.cwd() / "config"
    if cwd_candidate.is_dir() and (cwd_candidate / "engine.yaml").is_file():
        return cwd_candidate.resolve()

    # Relative to src/memorymesh/config.py -> project root -> config/
    module_root_candidate = Path(__file__).resolve().parents[2] / "config"
    if module_root_candidate.is_dir():
        return module_root_candidate.resolve()

    return cwd_candidate.resolve()


def load_yaml_file(file_path: Path | str) -> dict[str, Any]:
    """Read and parse a YAML file into a dictionary."""
    path = Path(file_path).resolve()
    if not path.is_file():
        raise ConfigFileNotFoundError(f"Configuration file not found: {path}")

    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ConfigParseError(f"Failed to parse YAML file '{path}': {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigParseError(
            f"Configuration file '{path}' must contain a mapping, got {type(data).__name__}"
        )

    return data


def load_engine_config(config_dir: Path | str | None = None) -> EngineConfig:
    """Load and validate engine.yaml, applying LLM_PROVIDER environment override if present."""
    base_dir = Path(config_dir) if config_dir is not None else get_default_config_dir()
    file_path = base_dir / "engine.yaml"
    raw_data = load_yaml_file(file_path)

    # Check for documented LLM_PROVIDER environment override
    env_provider = os.environ.get("LLM_PROVIDER")
    if env_provider and "llm" in raw_data and isinstance(raw_data["llm"], dict):
        raw_data["llm"]["provider"] = env_provider.strip().lower()

    try:
        return EngineConfig.model_validate(raw_data)
    except ValidationError as exc:
        raise ConfigValidationError(
            f"Invalid engine configuration in '{file_path}': {exc}"
        ) from exc


def load_sources_config(config_dir: Path | str | None = None) -> list[SourceConfig]:
    """Load and validate sources.yaml."""
    base_dir = Path(config_dir) if config_dir is not None else get_default_config_dir()
    file_path = base_dir / "sources.yaml"
    raw_data = load_yaml_file(file_path)

    try:
        parsed = SourcesConfigFile.model_validate(raw_data)
        return parsed.sources
    except ValidationError as exc:
        raise ConfigValidationError(
            f"Invalid sources configuration in '{file_path}': {exc}"
        ) from exc


def load_attributes_config(config_dir: Path | str | None = None) -> list[AttributeConfig]:
    """Load and validate attributes.yaml."""
    base_dir = Path(config_dir) if config_dir is not None else get_default_config_dir()
    file_path = base_dir / "attributes.yaml"
    raw_data = load_yaml_file(file_path)

    try:
        parsed = AttributesConfigFile.model_validate(raw_data)
        return parsed.attributes
    except ValidationError as exc:
        raise ConfigValidationError(
            f"Invalid attributes configuration in '{file_path}': {exc}"
        ) from exc


def load_entities_config(config_dir: Path | str | None = None) -> list[EntityConfig]:
    """Load and validate entities.yaml."""
    base_dir = Path(config_dir) if config_dir is not None else get_default_config_dir()
    file_path = base_dir / "entities.yaml"
    raw_data = load_yaml_file(file_path)

    try:
        parsed = EntitiesConfigFile.model_validate(raw_data)
        return parsed.entities
    except ValidationError as exc:
        raise ConfigValidationError(
            f"Invalid entities configuration in '{file_path}': {exc}"
        ) from exc


def load_app_config(config_dir: Path | str | None = None) -> AppConfig:
    """Load and validate all application configuration files."""
    base_dir = Path(config_dir) if config_dir is not None else get_default_config_dir()
    if not base_dir.is_dir():
        raise ConfigFileNotFoundError(f"Configuration directory not found: {base_dir}")

    return AppConfig(
        engine=load_engine_config(base_dir),
        sources=load_sources_config(base_dir),
        attributes=load_attributes_config(base_dir),
        entities=load_entities_config(base_dir),
    )
