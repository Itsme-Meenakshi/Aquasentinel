"""Project paths and settings, read from config/config.yaml and config/features.yaml."""
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load(name):
    with open(PROJECT_ROOT / "config" / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


CFG = _load("config.yaml")
FEATURES = _load("features.yaml")

TARGET_RAW = FEATURES["target"]
LEAKAGE_COLUMNS = FEATURES["leakage_columns"]
FEATURE_SETS = FEATURES["feature_sets"]


def path(key_or_rel: str) -> Path:
    """Resolve a path from config (e.g. 'data.cleaned_file') or a project-relative string."""
    if "." in key_or_rel and "/" not in key_or_rel and "\\" not in key_or_rel:
        section, key = key_or_rel.split(".", 1)
        return PROJECT_ROOT / CFG[section][key]
    return PROJECT_ROOT / key_or_rel
