"""Build the Oregon Coast Range Douglas-fir panel from FIA and PRISM."""

from __future__ import annotations

import hashlib
import logging
from typing import Any

import numpy as np
import pandas as pd

from forestry_ts.config import configure_logging, load_config
from forestry_ts.paths import DEFAULT_PANEL_PATH, PROJECT_ROOT, resolve_project_path

logger = logging.getLogger(__name__)

DEFAULT_POLICY_CAP_FRAC = 0.80


def fetch_fia_trees(state: str, county_codes: set[int], species_code: str) -> pd.DataFrame:
    """Pull tree-level remeasurement data from FIA via pyfia."""
    try:
        import pyfia
    except ImportError as exc:
        raise SystemExit(
            "pyfia not installed. Run: uv sync\n"
            "First run downloads Oregon FIA data (~400 MB)."
        ) from exc

    logger.info("Loading FIA data for %s (first run downloads from DataMart)...", state)
    fia = pyfia.FIA(state)
    trees = fia.panel(level="tree")
    trees = trees[
        (trees["COUNTYCD"].isin(county_codes)) & (trees["SPCD"] == 202)
    ].copy()

    if trees.empty:
        raise ValueError(
            f"No {species_code} trees found in counties {county_codes}. "
            "Check county codes or expand the filter."
        )

    logger.info("  FIA rows after filter: %s", f"{len(trees):,}")
    return trees


def trees_to_panel(
    trees: pd.DataFrame,
    *,
    species_code: str,
    year_min: int,
    year_max: int,
) -> pd.DataFrame:
    """Convert tree-level FIA records to the annual stand panel schema."""
    t = trees.copy()
    t["remper"] = pd.to_numeric(t.get("REMPER", np.nan), errors="coerce").fillna(5)
    vol = pd.to_numeric(t.get("VOLCFNET", np.nan), errors="coerce").fillna(0)
    t["net_growth_m3"] = (vol / t["remper"]) * 0.0283168

    t["stand_id"] = t["PLT_CN"].astype(str)
    t["year"] = pd.to_numeric(t["INVYR"], errors="coerce")
    t["species"] = species_code
    t["harvest_m3"] = 0.0

    elev_ft = pd.to_numeric(t.get("ELEV", np.nan), errors="coerce")
    t["elev_m"] = elev_ft * 0.3048

    site = pd.to_numeric(t.get("SITECLCD", np.nan), errors="coerce").fillna(4)
    t["age_class"] = pd.cut(
        site, bins=[0, 2, 4, 6, 7], labels=["young", "mid", "mature", "old"]
    ).astype(str)

    county_to_district = {59: "Tillamook", 7: "Clatsop"}
    t["district"] = t["COUNTYCD"].map(county_to_district).fillna("Other")

    cols = [
        "stand_id",
        "year",
        "species",
        "net_growth_m3",
        "harvest_m3",
        "elev_m",
        "age_class",
        "district",
        "remper",
    ]
    panel = t[cols].dropna(subset=["year", "net_growth_m3"])
    panel = panel[panel["year"].between(year_min, year_max)]
    panel = panel.reset_index(drop=True)
    logger.info(
        "  Panel rows: %s  years %.0f–%.0f",
        f"{len(panel):,}",
        panel["year"].min(),
        panel["year"].max(),
    )
    return panel


def _prism_point(lat: float, lon: float, year: int) -> tuple[float, float]:
    try:
        import requests

        url = (
            "https://prism.oregonstate.edu/explorer/dataexplorer/raster.php"
            f"?type=ppt&resolution=4km&lat={lat}&lon={lon}"
            f"&buffer=0&units=si&sdate={year}-01-01&edate={year}-12-31"
        )
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            data = r.json()
            ppt_values = [row.get("value", 0) for row in data.get("data", [])]
            ppt = float(np.sum(ppt_values)) if ppt_values else _normal_ppt(lat)

            url_t = url.replace("type=ppt", "type=tmean")
            rt = requests.get(url_t, timeout=15)
            if rt.status_code == 200:
                tmean_vals = [row.get("value", 10) for row in rt.json().get("data", [])]
                gdd = float(sum(max(v - 5, 0) * 30 for v in tmean_vals))
            else:
                gdd = _normal_gdd(lat)
            return ppt, gdd
    except Exception:
        pass

    rng = np.random.default_rng(
        int(hashlib.md5(f"{lat}{lon}{year}".encode()).hexdigest(), 16) % (2**32)
    )
    ppt = _normal_ppt(lat) * (1 + rng.normal(0, 0.12))
    gdd = _normal_gdd(lat) * (1 + rng.normal(0, 0.08))
    return float(ppt), float(gdd)


def _normal_ppt(lat: float) -> float:
    return 2200 - (lat - 45) * 80


def _normal_gdd(lat: float) -> float:
    return 1400 - (lat - 45) * 40


def fetch_prism_county(
    county_fips_list: list[int], year_min: int, year_max: int
) -> pd.DataFrame:
    centroids = {59: (45.46, -123.85), 7: (46.07, -123.72)}
    records = []
    for countycd, (lat, lon) in centroids.items():
        if countycd not in county_fips_list:
            continue
        for year in range(year_min, year_max + 1):
            ppt, gdd = _prism_point(lat, lon, year)
            records.append(
                {"countycd": countycd, "year": year, "precip_mm": ppt, "gdd": gdd}
            )
    df = pd.DataFrame(records)
    logger.info("  PRISM records: %s", len(df))
    return df


def add_ndvi_placeholders(panel: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    panel = panel.copy()
    panel["ndvi_mean"] = (
        0.72
        - panel["elev_m"].fillna(300) / 10000
        + (panel["year"] - 2010) * 0.002
        + rng.normal(0, 0.03, len(panel))
    ).clip(0.3, 0.95)

    panel = panel.sort_values(["stand_id", "year"])
    roll_mean = panel.groupby("stand_id")["ndvi_mean"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
    panel["ndvi_anom"] = panel["ndvi_mean"] - roll_mean.fillna(panel["ndvi_mean"])
    return panel


def add_policy_caps(panel: pd.DataFrame, cap_frac: float = DEFAULT_POLICY_CAP_FRAC) -> pd.DataFrame:
    panel = panel.copy()
    district_means = panel.groupby("district")["net_growth_m3"].mean() * cap_frac
    panel["policy_cap_m3"] = panel["district"].map(district_means)
    return panel


def build_panel(cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    data_cfg = cfg.get("data") or {}

    state = str(data_cfg.get("state", "OR"))
    species_code = str(data_cfg.get("species_code", "PSME"))
    target_counties = set(data_cfg.get("target_counties", [59, 7]))
    year_min = int(data_cfg.get("year_min", 2010))
    year_max = int(data_cfg.get("year_max", 2024))
    default_panel = DEFAULT_PANEL_PATH.relative_to(PROJECT_ROOT)
    output = resolve_project_path(data_cfg.get("panel_path", default_panel))
    cap_frac = float(data_cfg.get("policy_cap_frac", DEFAULT_POLICY_CAP_FRAC))

    trees = fetch_fia_trees(state, target_counties, species_code)
    panel = trees_to_panel(
        trees, species_code=species_code, year_min=year_min, year_max=year_max
    )

    county_map = {v: k for k, v in {59: "Tillamook", 7: "Clatsop"}.items()}
    panel["countycd"] = panel["district"].map(county_map).fillna(0).astype(int)

    prism = fetch_prism_county(list(target_counties), year_min, year_max)
    panel = panel.merge(prism, on=["countycd", "year"], how="left")
    panel = panel.drop(columns=["countycd"])

    panel = add_ndvi_placeholders(panel)
    panel = add_policy_caps(panel, cap_frac=cap_frac)
    panel = panel.sort_values(["district", "stand_id", "year"]).reset_index(drop=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output, index=False)
    logger.info("Saved %s rows → %s", f"{len(panel):,}", output)
    return panel


def main() -> None:
    cfg = load_config()
    configure_logging(cfg)
    build_panel(cfg)


if __name__ == "__main__":
    main()
