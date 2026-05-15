"""
run_forestry_figures.py

Loads coast_range_dfir_panel.parquet, trains a LightGBM model with
walk-forward validation, reconciles stand forecasts to district totals,
evaluates against a naive baseline, and writes four figures to ./figures/.

Usage
-----
    # Build parquet first (if not already done)
    python fetch_forestry_data.py

    # Then generate figures
    pip install lightgbm matplotlib
    python run_forestry_figures.py

Figures produced
----------------
    figures/01_district_actual_vs_predicted.png
    figures/02_sustainable_harvest_by_district.png
    figures/03_stand_predicted_vs_actual.png
    figures/04_wape_comparison.png
"""

from pathlib import Path

import signalplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import logging

def load_config(config_path=None):
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    if not config_path.exists():
        return {}
    with open(config_path) as _f:
        return _yaml.safe_load(_f) or {}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
PARQUET = Path("coast_range_dfir_panel.parquet")
FIGURES = Path("figures")
VAL_FROM = 2019

signalplot.apply(font_family='serif')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def wape(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.sum(np.abs(actual - forecast)) / (np.sum(np.abs(actual)) + 1e-9))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def run_model(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """
    Walk-forward LightGBM with cross-section or time-series mode.

    Returns (valid, cross_section_flag).
    """
    import lightgbm as lgb  # pip install lightgbm

    df = df.copy()
    df = df.query("species == 'PSME'").dropna(subset=["net_growth_m3"]).sort_values(["stand_id", "year"])

    n_per_stand = df.groupby("stand_id").size()
    cross_section = n_per_stand.max() <= 1

    skip = {"net_growth_m3", "harvest_m3", "stand_id", "species"}

    if cross_section:
        features = [
            c for c in df.columns
            if c not in skip
            and df[c].dtype.kind in "iufO"   # int, uint, float, object/category
        ]
    else:
        for col in ["net_growth_m3", "ndvi_mean", "precip_mm"]:
            if col in df.columns:
                df[f"{col}_lag1"] = df.groupby("stand_id")[col].shift(1)
                df[f"{col}_roll3"] = df.groupby("stand_id")[col].shift(1).rolling(3).mean()
        df = df.dropna(subset=["net_growth_m3_lag1"])
        features = [
            c for c in df.columns
            if c not in {"net_growth_m3", "harvest_m3", "year", "stand_id", "species", "district"}
        ]

    val_years = sorted(y for y in df["year"].unique() if y >= VAL_FROM)
    valid_parts = []

    for val_year in val_years:
        train = df[df.year < val_year].copy()
        valid_y = df[df.year == val_year].copy()
        if len(train) < 10 or len(valid_y) == 0:
            continue

        if "age_class" in train.columns:
            train["age_class"] = train["age_class"].astype("category")
            valid_y["age_class"] = valid_y["age_class"].astype("category")

        cat_feat = ["age_class"] if "age_class" in features else None
        model = lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, verbosity=-1
        )
        model.fit(train[features], train["net_growth_m3"], categorical_feature=cat_feat)
        valid_y["pred_growth_m3"] = model.predict(valid_y[features])
        pd.concat([valid_parts, valid_y])

    valid = pd.concat(valid_parts, ignore_index=True)
    logger.info(f"Mode: {'cross-section' if cross_section else 'time-series'}")
    logger.info(f"Validation rows: {len(valid):,}")
    return valid, cross_section


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------

def reconcile(valid: pd.DataFrame) -> pd.DataFrame:
    """Bottom-up: sum stand forecasts to district, apply policy cap."""
    stand_fc = (
        valid.groupby(["district", "year"])
        .agg(
            pred_growth_m3=("pred_growth_m3", "sum"),
            net_growth_m3=("net_growth_m3", "sum"),
            policy_cap_m3=("policy_cap_m3", "first"),
        )
        .reset_index()
    )
    stand_fc["sustainable_harvest_m3"] = stand_fc[["pred_growth_m3", "policy_cap_m3"]].min(axis=1)
    return stand_fc


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(valid: pd.DataFrame, df: pd.DataFrame, cross_section: bool) -> pd.DataFrame:
    """Attach naive baseline and compute WAPE."""
    if cross_section:
        district_prev = (
            df.groupby(["district", "year"])["net_growth_m3"].mean()
            .reset_index()
            .rename(columns={"net_growth_m3": "naive_growth_m3"})
        )
        district_prev["year"] = district_prev["year"] + 1
        valid = valid.merge(district_prev[["district", "year", "naive_growth_m3"]],
                            on=["district", "year"], how="left")
    else:
        prev = (
            df[["stand_id", "year", "net_growth_m3"]]
            .rename(columns={"net_growth_m3": "naive_growth_m3"})
        )
        prev["year"] = prev["year"] + 1
        valid = valid.merge(prev[["stand_id", "year", "naive_growth_m3"]],
                            on=["stand_id", "year"], how="left")
    return valid


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig1_district_actual_vs_predicted(district_fc: pd.DataFrame, out: Path, plot: bool = False) -> None:
    """Line chart: district-level actual vs predicted growth by year."""
    agg = (
        district_fc.groupby("year")[["net_growth_m3", "pred_growth_m3"]]
        .sum()
        .reset_index()
    )
    if plot:
        fig, ax = plt.subplots(figsize=tuple(config.get('output', {}).get('figsize', [8, 4])))
        ax.plot(agg["year"], agg["net_growth_m3"], marker="o", label="Actual")
        ax.plot(agg["year"], agg["pred_growth_m3"], marker="s", linestyle="--", label="Predicted")
        ax.set_title("District-level Actual vs Predicted Net Growth (state total)")
        ax.set_xlabel("Year")
        ax.set_ylabel("Net growth (m³)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
    logger.info(f"  Saved {out}")


def fig2_sustainable_harvest(district_fc: pd.DataFrame, out: Path, top_n: int = 10, plot: bool = False) -> None:
    """Bar chart: sustainable harvest vs predicted growth by district (latest year)."""
    latest_year = district_fc["year"].max()
    sub = (
        district_fc[district_fc["year"] == latest_year]
        .nlargest(top_n, "pred_growth_m3")
        .sort_values("pred_growth_m3", ascending=True)
    )
    y_pos = np.arange(len(sub))
    if plot:
        fig, ax = plt.subplots(figsize=(8, max(4, len(sub) * 0.5)))
        ax.barh(y_pos, sub["pred_growth_m3"], color="steelblue", alpha=0.7, label="Predicted growth")
        ax.barh(y_pos, sub["sustainable_harvest_m3"], color="seagreen", alpha=0.85, label="Sustainable harvest")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sub["district"])
        ax.set_xlabel("Volume (m³)")
        ax.set_title(f"Sustainable Harvest vs Predicted Growth by District ({latest_year})")
        ax.legend()
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
    logger.info(f"  Saved {out}")


def fig3_stand_scatter(valid: pd.DataFrame, out: Path, plot: bool = False) -> None:
    """Scatter: stand-level predicted vs actual (validation points)."""
    sub = valid.dropna(subset=["pred_growth_m3", "net_growth_m3"])
    lo = min(sub["net_growth_m3"].min(), sub["pred_growth_m3"].min()) * 0.9
    hi = max(sub["net_growth_m3"].max(), sub["pred_growth_m3"].max()) * 1.1
    if plot:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(sub["net_growth_m3"], sub["pred_growth_m3"],
                   alpha=0.25, s=10, color="steelblue")
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="1:1 line")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Actual net growth (m³)")
        ax.set_ylabel("Predicted net growth (m³)")
        ax.set_title("Stand-level Predicted vs Actual (validation)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
    logger.info(f"  Saved {out}")


def fig4_wape_comparison(wape_lgb: float, wape_naive: float, out: Path, plot: bool = False) -> None:
    """Bar chart: WAPE comparison between LightGBM and naive baseline."""
    labels = ["District prior\n(naive)", "LightGBM"]
    values = [wape_naive, wape_lgb]
    colors = ["#aec7e8", "#1f77b4"]
    if plot:
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(labels, values, color=colors, width=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=11)
        ax.set_ylabel("WAPE (lower is better)")
        ax.set_title("WAPE: District Prior Mean vs LightGBM")
        ax.set_ylim(0, max(values) * 1.25)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
    logger.info(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not PARQUET.exists():
        raise FileNotFoundError(
            f"{PARQUET} not found. Run fetch_forestry_data.py first."
        )

    FIGURES.mkdir(exist_ok=True)

    logger.info("Loading panel...")
    df = pd.read_parquet(PARQUET)
    logger.info(f"  {len(df):,} rows, {df['year'].min():.0f}–{df['year'].max():.0f}")

    logger.info("\nRunning model...")
    valid, cross_section = run_model(df)

    logger.info("\nReconciling to districts...")
    district_fc = reconcile(valid)

    logger.info("\nEvaluating...")
    valid = evaluate(valid, df, cross_section)
    eval_df = valid.dropna(subset=["naive_growth_m3"])
    wape_lgb = wape(eval_df["net_growth_m3"].values, eval_df["pred_growth_m3"].values)
    wape_naive = wape(eval_df["net_growth_m3"].values, eval_df["naive_growth_m3"].values)
    logger.info(f"  WAPE_lgb   = {wape_lgb:.3f}")
    logger.info(f"  WAPE_naive = {wape_naive:.3f}")

    logger.info("\nGenerating figures...")
    fig1_district_actual_vs_predicted(district_fc, FIGURES / "01_district_actual_vs_predicted.png")
    fig2_sustainable_harvest(district_fc, FIGURES / "02_sustainable_harvest_by_district.png")
    fig3_stand_scatter(valid, FIGURES / "03_stand_predicted_vs_actual.png")
    fig4_wape_comparison(wape_lgb, wape_naive, FIGURES / "04_wape_comparison.png")

    logger.info(f"\nDone. Figures in ./{FIGURES}/")


if __name__ == "__main__":
    main()
