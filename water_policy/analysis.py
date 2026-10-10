"""Core, testable analysis functions for the California water panel."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AnalysisWindow:
    """Inclusive pre/post windows around a policy year."""

    pre_start: int = 2014
    pre_end: int = 2017
    post_start: int = 2018
    post_end: int = 2020

    def __post_init__(self) -> None:
        if self.pre_start > self.pre_end:
            raise ValueError("pre_start must be no later than pre_end")
        if self.post_start > self.post_end:
            raise ValueError("post_start must be no later than post_end")
        if self.pre_end >= self.post_start:
            raise ValueError("pre and post windows must not overlap")


def assign_treatment(category: str) -> int:
    """Assign direct policy exposure by water-use category.

    SB 606/AB 1668 established an urban water-use efficiency framework.
    Public supply is therefore the more directly exposed category; irrigation
    is the less-directly exposed comparison category. This is an exploratory
    contrast, not proof of an untreated counterfactual.
    """

    normalized = category.strip().lower().replace("-", "_").replace(" ", "_")
    mapping = {"public_supply": 1, "irrigation": 0}
    if normalized not in mapping:
        raise ValueError(f"unknown water-use category: {category!r}")
    return mapping[normalized]


def compare_windows(
    panel: pd.DataFrame,
    window: AnalysisWindow,
    *,
    value_col: str = "total_use_mgd",
) -> pd.DataFrame:
    """Compute watershed-level pre/post annual means and changes.

    A watershed is retained only if every year in both requested windows is
    present exactly once. This prevents a partially observed window from being
    silently compared with a complete one.
    """

    required = {"watershed", "year", value_col}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"panel is missing columns: {sorted(missing)}")

    years = list(range(window.pre_start, window.pre_end + 1)) + list(
        range(window.post_start, window.post_end + 1)
    )
    subset = panel.loc[panel["year"].isin(years), ["watershed", "year", value_col]].copy()
    if subset.duplicated(["watershed", "year"]).any():
        raise ValueError("panel must contain one row per watershed-year")

    expected = len(years)
    complete = subset.groupby("watershed")["year"].nunique().eq(expected)
    subset = subset[subset["watershed"].isin(complete[complete].index)]
    if subset.empty:
        return pd.DataFrame(
            columns=["watershed", "pre_mean_mgd", "post_mean_mgd", "absolute_change_mgd", "percent_change"]
        )

    pre = (
        subset[subset["year"].between(window.pre_start, window.pre_end)]
        .groupby("watershed")[value_col]
        .mean()
        .rename("pre_mean_mgd")
    )
    post = (
        subset[subset["year"].between(window.post_start, window.post_end)]
        .groupby("watershed")[value_col]
        .mean()
        .rename("post_mean_mgd")
    )
    result = pd.concat([pre, post], axis=1).reset_index()
    result["absolute_change_mgd"] = result["post_mean_mgd"] - result["pre_mean_mgd"]
    result["percent_change"] = np.where(
        result["pre_mean_mgd"].ne(0),
        100 * result["absolute_change_mgd"] / result["pre_mean_mgd"],
        np.nan,
    )
    return result.sort_values("watershed").reset_index(drop=True)


def bootstrap_median_ci(
    values: pd.Series | np.ndarray,
    *,
    confidence: float = 0.95,
    n_bootstrap: int = 20_000,
    seed: int = 2026,
) -> tuple[float, float]:
    """Percentile bootstrap interval for a median, resampling watersheds."""

    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        raise ValueError("cannot bootstrap an empty sample")
    rng = np.random.default_rng(seed)
    samples = rng.choice(clean, size=(n_bootstrap, clean.size), replace=True)
    medians = np.median(samples, axis=1)
    alpha = 1 - confidence
    return tuple(np.quantile(medians, [alpha / 2, 1 - alpha / 2]))
