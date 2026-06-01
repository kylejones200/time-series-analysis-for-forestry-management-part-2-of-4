"""Train model, reconcile districts, and write validation figures."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import signalplot

from forestry_ts import __version__
from forestry_ts.config import configure_logging, load_config
from forestry_ts.figures import (
    fig1_district_actual_vs_predicted,
    fig2_sustainable_harvest,
    fig3_stand_scatter,
    fig4_wape_comparison,
)
from forestry_ts.model import evaluate, reconcile, run_model, wape
from forestry_ts.paths import DEFAULT_PANEL_PATH, PROJECT_ROOT, resolve_project_path

logger = logging.getLogger(__name__)


def run(config_path: Path | str | None = None) -> dict[str, Any]:
    cfg = load_config(config_path)
    configure_logging(cfg)
    signalplot.apply(font_family="serif")
    data_cfg = cfg.get("data") or {}
    panel_path = resolve_project_path(
        data_cfg.get("panel_path", DEFAULT_PANEL_PATH.relative_to(PROJECT_ROOT))
    )
    if not panel_path.is_file():
        raise FileNotFoundError(f"{panel_path} not found. Run: uv run forestry-fetch")

    out_cfg = cfg.get("output") or {}
    figures_dir = resolve_project_path(out_cfg.get("figures_dir", "outputs/figures"))
    figures_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Loading panel from %s", panel_path)
    df = pd.read_parquet(panel_path)
    logger.info(
        "%s rows, %.0f–%.0f",
        f"{len(df):,}",
        df["year"].min(),
        df["year"].max(),
    )
    logger.info("Running model...")
    valid, cross_section = run_model(df, cfg)
    logger.info("Reconciling to districts...")
    district_fc = reconcile(valid)
    logger.info("Evaluating...")
    valid = evaluate(valid, df, cross_section)
    eval_df = valid.dropna(subset=["naive_growth_m3"])
    wape_lgb = wape(eval_df["net_growth_m3"].values, eval_df["pred_growth_m3"].values)
    wape_naive = wape(eval_df["net_growth_m3"].values, eval_df["naive_growth_m3"].values)
    logger.info("WAPE_lgb   = %.3f", wape_lgb)
    logger.info("WAPE_naive = %.3f", wape_naive)
    logger.info("Generating figures...")
    fig1_district_actual_vs_predicted(
        district_fc, figures_dir / "01_district_actual_vs_predicted.png", cfg
    )
    fig2_sustainable_harvest(
        district_fc, figures_dir / "02_sustainable_harvest_by_district.png", cfg
    )
    fig3_stand_scatter(valid, figures_dir / "03_stand_predicted_vs_actual.png", cfg)
    fig4_wape_comparison(wape_lgb, wape_naive, figures_dir / "04_wape_comparison.png", cfg)
    logger.info("Done. Figures in %s", figures_dir)
    return {
        "wape_lgb": wape_lgb,
        "wape_naive": wape_naive,
        "figures_dir": str(figures_dir.relative_to(PROJECT_ROOT)),
        "cross_section": cross_section,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Forestry panel forecasting and figure generation (Part 2)"
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to config.yaml")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
