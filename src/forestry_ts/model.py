"""Walk-forward LightGBM, district reconciliation, and evaluation."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def wape(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.sum(np.abs(actual - forecast)) / (np.sum(np.abs(actual)) + 1e-9))


def run_model(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, bool]:
    """Walk-forward LightGBM with cross-section or time-series mode."""
    import lightgbm as lgb

    model_cfg = cfg.get("model") or {}
    val_from = int(model_cfg.get("val_from", 2019))
    species_filter = str(model_cfg.get("species_filter", "PSME"))

    df = df.copy()
    df = (
        df[df["species"] == species_filter]
        .dropna(subset=["net_growth_m3"])
        .sort_values(["stand_id", "year"])
    )

    n_per_stand = df.groupby("stand_id").size()
    cross_section = n_per_stand.max() <= 1
    skip = {"net_growth_m3", "harvest_m3", "stand_id", "species"}

    if cross_section:
        features = [
            c
            for c in df.columns
            if c not in skip and df[c].dtype.kind in "iufO"
        ]
    else:
        for col in ["net_growth_m3", "ndvi_mean", "precip_mm"]:
            if col in df.columns:
                df[f"{col}_lag1"] = df.groupby("stand_id")[col].shift(1)
                df[f"{col}_roll3"] = (
                    df.groupby("stand_id")[col].shift(1).rolling(3).mean()
                )
        df = df.dropna(subset=["net_growth_m3_lag1"])
        features = [
            c
            for c in df.columns
            if c
            not in {
                "net_growth_m3",
                "harvest_m3",
                "year",
                "stand_id",
                "species",
                "district",
            }
        ]

    n_estimators = int(model_cfg.get("n_estimators", 500))
    learning_rate = float(model_cfg.get("learning_rate", 0.05))

    val_years = sorted(y for y in df["year"].unique() if y >= val_from)
    valid_parts: list[pd.DataFrame] = []

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
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            verbosity=-1,
        )
        model.fit(train[features], train["net_growth_m3"], categorical_feature=cat_feat)
        valid_y["pred_growth_m3"] = model.predict(valid_y[features])
        valid_parts.append(valid_y)

    valid = pd.concat(valid_parts, ignore_index=True)
    logger.info("Mode: %s", "cross-section" if cross_section else "time-series")
    logger.info("Validation rows: %s", f"{len(valid):,}")
    return valid, cross_section


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
    stand_fc["sustainable_harvest_m3"] = stand_fc[
        ["pred_growth_m3", "policy_cap_m3"]
    ].min(axis=1)
    return stand_fc


def evaluate(
    valid: pd.DataFrame, df: pd.DataFrame, cross_section: bool
) -> pd.DataFrame:
    """Attach naive baseline for WAPE comparison."""
    if cross_section:
        district_prev = (
            df.groupby(["district", "year"])["net_growth_m3"]
            .mean()
            .reset_index()
            .rename(columns={"net_growth_m3": "naive_growth_m3"})
        )
        district_prev["year"] = district_prev["year"] + 1
        valid = valid.merge(
            district_prev[["district", "year", "naive_growth_m3"]],
            on=["district", "year"],
            how="left",
        )
    else:
        prev = df[["stand_id", "year", "net_growth_m3"]].rename(
            columns={"net_growth_m3": "naive_growth_m3"}
        )
        prev["year"] = prev["year"] + 1
        valid = valid.merge(
            prev[["stand_id", "year", "naive_growth_m3"]],
            on=["stand_id", "year"],
            how="left",
        )
    return valid
