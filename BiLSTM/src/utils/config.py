from pathlib import Path
from typing import Any

import yaml

REQUIRED_SECTIONS = {
    "baseline": ["model", "data", "training"],
    "vat": ["model", "data", "training", "vat"],
    "lm_pretrain": ["lm", "data", "training"],
}

def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if config is None:
        raise ValueError(f"Config file is empty: {path}")
    if not isinstance(config, dict):
        raise TypeError(f"Config must be a dictionary, got: {type(config)}")
    
    return config


def validate_config_sections(config: dict[str, Any], config_type: str) -> None:
    if config_type not in REQUIRED_SECTIONS:
        raise ValueError(f"Unknown config_type: {config_type}")

    required_sections = REQUIRED_SECTIONS[config_type]
    missing_sections = [section for section in required_sections if section not in config]

    if missing_sections:
        raise ValueError(f"Missing required sections: {missing_sections}")


def load_and_validate_config(config_path: str | Path, config_type: str) -> dict[str, Any]:
    config = load_yaml_config(config_path)
    validate_config_sections(config, config_type)
    return config