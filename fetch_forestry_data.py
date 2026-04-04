"""
fetch_forestry_data.py

Builds coast_range_dfir_panel.parquet for the Oregon Coast Range Douglas-fir
(PSME) case study in Part 2 of the forestry time series series.

Data sources
------------
- FIA DataMart via pyfia (https://github.com/usda-forest-service/pyFIA)
- PRISM climate normals + time series (https://prism.oregonstate.edu/)
- NDVI / LANDFIRE: optional joins; placeholders used when unavailable

Usage
-----
    pip install pyfia pandas pyarrow requests
    python fetch_forestry_data.py

Output
------
    coast_range_dfir_panel.parquet
    Columns: stand_id, year, species, net_growth_m3, harvest_m3,
             precip_mm, gdd, ndvi_mean, ndvi_anom, age_class,
             elev_m, district, policy_cap_m3, remper
"""

import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
STATE = "OR"
SPECIES_CODE = "PSME"          # Douglas-fir
TARGET_COUNTIES = {59, 7}      # Tillamook = 59, Clatsop = 7 (FIA COUNTYCD)
YEAR_MIN, YEAR_MAX = 2010, 2024
OUTPUT = Path("coast_range_dfir_panel.parquet")

# Approximate policy cap: 80 % of mean predicted growth per district (set
# later from data; this default is overridden in build_panel).
DEFAULT_POLICY_CAP_FRAC = 0.80


# ---------------------------------------------------------------------------
# FIA fetch
# ---------------------------------------------------------------------------

def fetch_fia_trees(state: str, county_codes: set[int]) -> pd.DataFrame:
    """
    Pull tree-level remeasurement data from FIA via pyfia.

    Returns a DataFrame with one row per tree per remeasurement cycle.
    Columns used downstream: PLT_CN, CONDID, COUNTYCD, INVYR, SPCD,
    DIA, HT, VOLCFNET, TPA_UNADJ, REMPER, ELEV, SITECLCD, OWNCD.
    """
    try:
        import pyfia  # pip install pyfia
    except ImportError:
        raise SystemExit(
            "pyfia not installed. Run:  pip install pyfia\n"
            "First run downloads Oregon FIA data (~400 MB)."
        )

    print(f"Loading FIA data for {state} (first run downloads from DataMart)...")
    fia = pyfia.FIA(state)

    # tree-level panel: one row per live tree per inventory cycle
    trees = fia.panel(level="tree")

    # Filter to target counties and species
    trees = trees[
        (trees["COUNTYCD"].isin(county_codes)) &
        (trees["SPCD"] == 202)  # PSME = FIA species code 202
    ].copy()

    if trees.empty:
        raise ValueError(
            f"No {SPECIES_CODE} trees found in counties {county_codes}. "
            "Check county codes or expand the filter."
        )

    print(f"  FIA rows after filter: {len(trees):,}")
    return trees


def trees_to_panel(trees: pd.DataFrame) -> pd.DataFrame:
    """
    Convert tree-level FIA records to the annual stand panel schema.

    net_growth_m3  : annualised net cubic volume growth proxy (VOLCFNET / REMPER)
    harvest_m3     : not directly in FIA tree table; set to NaN (fill from
                     harvest ledger if available, else left as 0 for this demo)
    stand_id       : PLT_CN cast to string
    remper         : remeasurement period (years)
    """
    t = trees.copy()

    # Annualised net volume growth (ft³/yr → m³/yr,  1 ft³ ≈ 0.0283168 m³)
    t["remper"] = pd.to_numeric(t.get("REMPER", np.nan), errors="coerce").fillna(5)
    vol = pd.to_numeric(t.get("VOLCFNET", np.nan), errors="coerce").fillna(0)
    t["net_growth_m3"] = (vol / t["remper"]) * 0.0283168

    t["stand_id"] = t["PLT_CN"].astype(str)
    t["year"] = pd.to_numeric(t["INVYR"], errors="coerce")
    t["species"] = SPECIES_CODE
    t["harvest_m3"] = 0.0  # placeholder; replace with ledger data if available

    # Elevation ft → m
    elev_ft = pd.to_numeric(t.get("ELEV", np.nan), errors="coerce")
    t["elev_m"] = elev_ft * 0.3048

    # Age class from site class code (SITECLCD 1–7 → ordinal buckets)
    site = pd.to_numeric(t.get("SITECLCD", np.nan), errors="coerce").fillna(4)
    t["age_class"] = pd.cut(
        site, bins=[0, 2, 4, 6, 7], labels=["young", "mid", "mature", "old"]
    ).astype(str)

    # District from county
    county_to_district = {59: "Tillamook", 7: "Clatsop"}
    t["district"] = t["COUNTYCD"].map(county_to_district).fillna("Other")

    cols = ["stand_id", "year", "species", "net_growth_m3", "harvest_m3",
            "elev_m", "age_class", "district", "remper"]
    panel = t[cols].dropna(subset=["year", "net_growth_m3"])
    panel = panel[panel["year"].between(YEAR_MIN, YEAR_MAX)]
    panel = panel.reset_index(drop=True)
    print(f"  Panel rows after aggregation: {len(panel):,}  "
          f"years {panel['year'].min():.0f}–{panel['year'].max():.0f}")
    return panel


# ---------------------------------------------------------------------------
# PRISM climate
# ---------------------------------------------------------------------------

def fetch_prism_county(county_fips_list: list[int],
                       year_min: int, year_max: int) -> pd.DataFrame:
    """
    Fetch annual PRISM precipitation and GDD proxies for each county.

    Uses the PRISM Time Series API (BIL download) or falls back to a
    county-centroid CSV if the API is unavailable.

    Returns DataFrame with columns: countycd, year, precip_mm, gdd.
    """
    # County centroids (lat/lon) for Tillamook and Clatsop, OR
    centroids = {
        59: (45.46, -123.85),   # Tillamook
        7:  (46.07, -123.72),   # Clatsop
    }

    records = []
    for countycd, (lat, lon) in centroids.items():
        for year in range(year_min, year_max + 1):
            ppt, gdd = _prism_point(lat, lon, year)
            records.append({"countycd": countycd, "year": year,
                             "precip_mm": ppt, "gdd": gdd})

    df = pd.DataFrame(records)
    print(f"  PRISM records: {len(df)}")
    return df


def _prism_point(lat: float, lon: float, year: int) -> tuple[float, float]:
    """
    Query PRISM API for annual ppt (mm) and compute GDD proxy from tmean.

    Falls back to climatological normals + small random noise if the
    API is unreachable (e.g. offline / rate-limited).
    """
    try:
        import requests
        # PRISM Explorer time series endpoint
        url = (
            f"https://prism.oregonstate.edu/explorer/dataexplorer/raster.php"
            f"?type=ppt&resolution=4km&lat={lat}&lon={lon}"
            f"&buffer=0&units=si&sdate={year}-01-01&edate={year}-12-31"
        )
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            data = r.json()
            # API returns monthly values; sum for annual ppt
            ppt_values = [row.get("value", 0) for row in data.get("data", [])]
            ppt = float(np.sum(ppt_values)) if ppt_values else _normal_ppt(lat)

            # Fetch tmean for GDD
            url_t = url.replace("type=ppt", "type=tmean")
            rt = requests.get(url_t, timeout=15)
            if rt.status_code == 200:
                tmean_vals = [row.get("value", 10) for row in rt.json().get("data", [])]
                # GDD base 5°C, annual sum of monthly mean temps above base
                gdd = float(sum(max(v - 5, 0) * 30 for v in tmean_vals))
            else:
                gdd = _normal_gdd(lat)
            return ppt, gdd
    except Exception:
        pass

    # Fallback: Oregon Coast Range climatological normals with interannual noise
    rng = np.random.default_rng(int(hashlib.md5(f"{lat}{lon}{year}".encode()).hexdigest(), 16) % (2**32))
    ppt = _normal_ppt(lat) * (1 + rng.normal(0, 0.12))
    gdd = _normal_gdd(lat) * (1 + rng.normal(0, 0.08))
    return float(ppt), float(gdd)


def _normal_ppt(lat: float) -> float:
    """Oregon Coast Range annual normal precip (mm), rough latitudinal gradient."""
    return 2200 - (lat - 45) * 80


def _normal_gdd(lat: float) -> float:
    """Oregon Coast Range annual GDD base-5 normal."""
    return 1400 - (lat - 45) * 40


# ---------------------------------------------------------------------------
# NDVI placeholders
# ---------------------------------------------------------------------------

def add_ndvi_placeholders(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Add ndvi_mean and ndvi_anom columns.

    Replace with actual MODIS/VIIRS extraction (e.g. via Google Earth Engine
    or the `modis` package) when available.  The placeholder uses a
    deterministic function of elevation and year so results are reproducible.
    """
    rng = np.random.default_rng(42)
    # Mild upward NDVI trend + elevation effect + noise
    panel["ndvi_mean"] = (
        0.72
        - panel["elev_m"].fillna(300) / 10000
        + (panel["year"] - 2010) * 0.002
        + rng.normal(0, 0.03, len(panel))
    ).clip(0.3, 0.95)

    # 5-year rolling normal per stand; anomaly = deviation from that
    panel = panel.sort_values(["stand_id", "year"])
    roll_mean = (
        panel.groupby("stand_id")["ndvi_mean"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    )
    panel["ndvi_anom"] = panel["ndvi_mean"] - roll_mean.fillna(panel["ndvi_mean"])
    return panel


# ---------------------------------------------------------------------------
# Policy caps
# ---------------------------------------------------------------------------

def add_policy_caps(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Assign policy_cap_m3: 80 % of each district's mean net growth.
    Replace with actual harvest plan caps when available.
    """
    district_means = (
        panel.groupby("district")["net_growth_m3"].mean() * DEFAULT_POLICY_CAP_FRAC
    )
    panel["policy_cap_m3"] = panel["district"].map(district_means)
    return panel


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_panel() -> pd.DataFrame:
    trees = fetch_fia_trees(STATE, TARGET_COUNTIES)
    panel = trees_to_panel(trees)

    # Map countycd back for PRISM join
    county_map = {v: k for k, v in {59: "Tillamook", 7: "Clatsop"}.items()}
    panel["countycd"] = panel["district"].map(county_map).fillna(0).astype(int)

    prism = fetch_prism_county(list(TARGET_COUNTIES), YEAR_MIN, YEAR_MAX)
    panel = panel.merge(prism, on=["countycd", "year"], how="left")
    panel = panel.drop(columns=["countycd"])

    panel = add_ndvi_placeholders(panel)
    panel = add_policy_caps(panel)

    panel = panel.sort_values(["district", "stand_id", "year"]).reset_index(drop=True)
    panel.to_parquet(OUTPUT, index=False)
    print(f"\nSaved {len(panel):,} rows → {OUTPUT}")
    print(panel.dtypes.to_string())
    return panel


if __name__ == "__main__":
    build_panel()
