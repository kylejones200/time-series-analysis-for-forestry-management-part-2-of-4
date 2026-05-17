"""Article figures: district totals, harvest caps, stand scatter, WAPE."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _figsize(cfg: dict[str, Any], default: tuple[float, float] = (8.0, 4.0)) -> tuple[float, float]:
    raw = (cfg.get("output") or {}).get("figsize", list(default))
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        return float(raw[0]), float(raw[1])
    return default


def fig1_district_actual_vs_predicted(
    district_fc: pd.DataFrame, out: Path, cfg: dict[str, Any]
) -> None:
    agg = (
        district_fc.groupby("year")[["net_growth_m3", "pred_growth_m3"]]
        .sum()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=_figsize(cfg))
    ax.plot(agg["year"], agg["net_growth_m3"], marker="o", label="Actual")
    ax.plot(
        agg["year"],
        agg["pred_growth_m3"],
        marker="s",
        linestyle="--",
        label="Predicted",
    )
    ax.set_title("District-level Actual vs Predicted Net Growth (state total)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Net growth (m³)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=int((cfg.get("output") or {}).get("figure_dpi", 100)))
    plt.close(fig)
    logger.info("  Saved %s", out)


def fig2_sustainable_harvest(
    district_fc: pd.DataFrame, out: Path, cfg: dict[str, Any], top_n: int = 10
) -> None:
    latest_year = district_fc["year"].max()
    sub = (
        district_fc[district_fc["year"] == latest_year]
        .nlargest(top_n, "pred_growth_m3")
        .sort_values("pred_growth_m3", ascending=True)
    )
    y_pos = np.arange(len(sub))
    fig, ax = plt.subplots(figsize=(8, max(4, len(sub) * 0.5)))
    ax.barh(
        y_pos,
        sub["pred_growth_m3"],
        color="steelblue",
        alpha=0.7,
        label="Predicted growth",
    )
    ax.barh(
        y_pos,
        sub["sustainable_harvest_m3"],
        color="seagreen",
        alpha=0.85,
        label="Sustainable harvest",
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sub["district"])
    ax.set_xlabel("Volume (m³)")
    ax.set_title(f"Sustainable Harvest vs Predicted Growth by District ({latest_year})")
    ax.legend()
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=int((cfg.get("output") or {}).get("figure_dpi", 100)))
    plt.close(fig)
    logger.info("  Saved %s", out)


def fig3_stand_scatter(valid: pd.DataFrame, out: Path, cfg: dict[str, Any]) -> None:
    sub = valid.dropna(subset=["pred_growth_m3", "net_growth_m3"])
    lo = min(sub["net_growth_m3"].min(), sub["pred_growth_m3"].min()) * 0.9
    hi = max(sub["net_growth_m3"].max(), sub["pred_growth_m3"].max()) * 1.1
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        sub["net_growth_m3"],
        sub["pred_growth_m3"],
        alpha=0.25,
        s=10,
        color="steelblue",
    )
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="1:1 line")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Actual net growth (m³)")
    ax.set_ylabel("Predicted net growth (m³)")
    ax.set_title("Stand-level Predicted vs Actual (validation)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=int((cfg.get("output") or {}).get("figure_dpi", 100)))
    plt.close(fig)
    logger.info("  Saved %s", out)


def fig4_wape_comparison(
    wape_lgb: float, wape_naive: float, out: Path, cfg: dict[str, Any]
) -> None:
    labels = ["District prior\n(naive)", "LightGBM"]
    values = [wape_naive, wape_lgb]
    colors = ["#aec7e8", "#1f77b4"]
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, values, color=colors, width=0.5)
    for bar, val in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    ax.set_ylabel("WAPE (lower is better)")
    ax.set_title("WAPE: District Prior Mean vs LightGBM")
    ax.set_ylim(0, max(values) * 1.25)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=int((cfg.get("output") or {}).get("figure_dpi", 100)))
    plt.close(fig)
    logger.info("  Saved %s", out)
