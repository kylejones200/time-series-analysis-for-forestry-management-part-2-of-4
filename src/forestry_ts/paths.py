"""Repository root and default paths for data, config, and outputs."""

from pathlib import Path

# src/forestry_ts/paths.py -> repo root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_PANEL_PATH = DATA_DIR / "coast_range_dfir_panel.parquet"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"


def resolve_project_path(rel: str | Path) -> Path:
    """Resolve a config-relative path against the repository root."""
    path = Path(rel)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path
