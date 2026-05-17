# Time Series Analysis for Forestry Management (Part 2 of 4)

Medium: [Time Series Analysis for Forestry Management — Part 2 of 4](https://medium.com/@kyle-t-jones/time-series-analysis-for-forestry-management-part-2-of-4-81d9f34e2394)

Oregon Coast Range Douglas-fir (PSME) case study. Builds a multi-stand annual panel from public FIA and PRISM data, trains a LightGBM walk-forward model, reconciles stand forecasts to district totals, and evaluates against a district prior mean baseline.

## Quick start

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run forestry-fetch   # builds data/coast_range_dfir_panel.parquet (~first run downloads OR FIA data)
uv run forestry-run     # trains model, reconciles, evaluates, writes outputs/figures/
```

## Project layout

```
config.yaml              # data paths, model hyperparameters, output settings
pyproject.toml / uv.lock
src/forestry_ts/         # data fetch, model, figures, CLI
data/                    # generated panel parquet (gitignored)
outputs/figures/         # generated plots (gitignored)
article.md
```

## Output figures

- `outputs/figures/01_district_actual_vs_predicted.png` — district-level actual vs predicted growth by year
- `outputs/figures/02_sustainable_harvest_by_district.png` — sustainable harvest vs predicted growth (latest year, top districts)
- `outputs/figures/03_stand_predicted_vs_actual.png` — stand-level predicted vs actual scatter (validation points)
- `outputs/figures/04_wape_comparison.png` — WAPE: district prior mean baseline vs LightGBM

## Data sources

- FIA DataMart (USDA Forest Service) — https://www.fia.fs.usda.gov/
- PRISM Climate (Oregon State) — https://prism.oregonstate.edu/
- MODIS/VIIRS NDVI — optional; placeholders used when unavailable

## Results (from article run)

Panel: 7,115 rows (tree-level), Oregon Coast Range, 2011–2022. Validation years 2019–2022. Cross-section mode (one remeasurement per tree).

| Model | WAPE |
|-------|------|
| District prior mean (naive) | 0.92 |
| LightGBM | 1.15 |

The district prior mean baseline beats the model at tree-level resolution — consistent with sparse FIA panels where simple persistence is hard to beat without aggregation or richer lag features.

## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).
