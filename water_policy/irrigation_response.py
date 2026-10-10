"""Pre-specified HUC12 irrigation response analysis.

Run from the project directory with::

    python -m water_policy.irrigation_response
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import WATERSHEDS, load_weather


SPEC_SHA256 = "4e4a8b7cc108b623f415eab1829e35c6af785d4c3ddacea976935f2d478e7e37"
PRIMARY_MIN_YEAR = 2000
PRIMARY_MAX_YEAR = 2020
PRIMARY_DROUGHT_CUTOFF = -1.0


def _california_columns(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        header = next(csv.reader(handle))
    return ["Year"] + [c for c in header if c[:4] in WATERSHEDS]


def _load_annual_wide(path: Path, value_name: str) -> pd.DataFrame:
    columns = _california_columns(path)
    wide = pd.read_csv(path, usecols=columns)
    values = wide.drop(columns="Year").apply(pd.to_numeric, errors="coerce").mask(lambda x: x.eq(999))
    values.insert(0, "year", wide["Year"].astype(int))
    return values.melt(id_vars="year", var_name="huc12", value_name=value_name)


def _load_monthly_irrigation_as_annual(path: Path) -> pd.DataFrame:
    columns = _california_columns(path)
    columns.insert(1, "Month")
    monthly = pd.read_csv(path, usecols=columns)
    return annualize_monthly_huc12(monthly).melt(
        id_vars="year", var_name="huc12", value_name="irrigation_mgd"
    )


def clean_water_values(values: pd.DataFrame) -> pd.DataFrame:
    """Coerce source values to numeric and replace the 999 missing sentinel."""

    numeric = values.apply(pd.to_numeric, errors="coerce")
    return numeric.mask(numeric.eq(999))


def annualize_monthly_huc12(monthly: pd.DataFrame) -> pd.DataFrame:
    """Convert monthly mean MGD to calendar-day-weighted annual mean MGD."""

    required = {"Year", "Month"}
    if not required.issubset(monthly.columns):
        raise ValueError("monthly data must include Year and Month")
    values = clean_water_values(monthly.drop(columns=["Year", "Month"]))
    days = pd.to_datetime(
        {"year": monthly["Year"], "month": monthly["Month"], "day": 1}
    ).dt.days_in_month
    annual = values.mul(days, axis=0).groupby(monthly["Year"]).sum(min_count=1)
    annual = annual.div(days.groupby(monthly["Year"]).sum(), axis=0)
    complete_months = values.notna().groupby(monthly["Year"]).sum().eq(12)
    annual = annual.where(complete_months)
    annual.insert(0, "year", annual.index.astype(int))
    annual = annual.reset_index(drop=True)
    return annual


def build_huc12_panel(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the full positive-pair panel and an availability audit."""

    irrigation = _load_monthly_irrigation_as_annual(
        data_dir / "Irr_CU_HUC12_Tot_monthly_2000_2020.csv"
    )
    precipitation = _load_annual_wide(
        data_dir / "PPTeff_HUC12_Tot_annual_2000_2020.csv", "effective_precipitation_mgd"
    )
    panel = irrigation.merge(precipitation, on=["huc12", "year"], validate="one_to_one")
    panel["huc4"] = panel["huc12"].str[:4]
    panel["watershed"] = panel["huc4"].map(WATERSHEDS)

    weather = load_weather(data_dir)[["huc4", "year", "temperature_f", "temperature_station_count"]]
    panel = panel.merge(weather, on=["huc4", "year"], how="left", validate="many_to_one")
    panel["positive_pair"] = panel["irrigation_mgd"].gt(0) & panel["effective_precipitation_mgd"].gt(0)

    availability = pd.DataFrame(
        [
            {
                "geography_available": "HUC12",
                "huc12_columns_in_nine_huc4_regions": panel["huc12"].nunique(),
                "years": f"{panel.year.min()}-{panel.year.max()}",
                "total_unit_year_rows": len(panel),
                "paired_nonmissing_rows": int(
                    panel[["irrigation_mgd", "effective_precipitation_mgd"]].notna().all(axis=1).sum()
                ),
                "paired_positive_rows": int(panel["positive_pair"].sum()),
                "balanced_positive_huc12_units": int(
                    panel.groupby("huc12")["positive_pair"].all().sum()
                ),
                "huc12_units_with_at_least_15_positive_years": int(
                    panel.groupby("huc12")["positive_pair"].sum().ge(15).sum()
                ),
                "temperature_geography": "HUC4 (3 NOAA stations where coverage permits)",
            }
        ]
    )
    return panel, availability


def source_identity_check(data_dir: Path, annualized_irrigation: pd.DataFrame) -> pd.DataFrame:
    """Verify that the ambiguous annual file is ETa, not irrigation CU."""

    ambiguous = _load_annual_wide(data_dir / "Irrigation_wateruse.csv", "ambiguous_annual_mgd")
    precipitation = _load_annual_wide(
        data_dir / "PPTeff_HUC12_Tot_annual_2000_2020.csv", "effective_precipitation_mgd"
    )
    check = annualized_irrigation.merge(ambiguous, on=["huc12", "year"], validate="one_to_one")
    check = check.merge(precipitation, on=["huc12", "year"], validate="one_to_one").dropna()
    check["identity_error_mgd"] = (
        check["ambiguous_annual_mgd"]
        - check["effective_precipitation_mgd"]
        - check["irrigation_mgd"]
    )
    relative = check["identity_error_mgd"].abs() / check["irrigation_mgd"].abs().clip(lower=1e-12)
    return pd.DataFrame(
        [
            {
                "identity_test": "Irrigation_wateruse - PPTeff = day-weighted monthly irrigation CU",
                "matched_unit_years": len(check),
                "mean_identity_error_mgd": check["identity_error_mgd"].mean(),
                "max_absolute_identity_error_mgd": check["identity_error_mgd"].abs().max(),
                "max_absolute_relative_error": relative.max(),
                "conclusion": "Irrigation_wateruse.csv is total crop ET, not irrigation consumptive use",
            }
        ]
    )


def select_balanced_positive(panel: pd.DataFrame, start: int = 2000, end: int = 2020) -> pd.DataFrame:
    subset = panel[panel["year"].between(start, end)].copy()
    expected = end - start + 1
    eligible = subset.groupby("huc12").agg(rows=("year", "nunique"), positive=("positive_pair", "sum"))
    units = eligible.index[(eligible["rows"].eq(expected)) & (eligible["positive"].eq(expected))]
    selected = subset[subset["huc12"].isin(units)].copy()
    return prepare_variables(selected)


def select_unbalanced(panel: pd.DataFrame, minimum_positive_years: int = 15) -> pd.DataFrame:
    counts = panel.groupby("huc12")["positive_pair"].sum()
    units = counts[counts >= minimum_positive_years].index
    selected = panel[panel["huc12"].isin(units) & panel["positive_pair"]].copy()
    return prepare_variables(selected)


def standardized_precipitation_proxy(values: pd.Series) -> pd.Series:
    """Gamma-CDF-to-normal transform used by the SPI construction."""

    x = values.to_numpy(dtype=float)
    if len(x) < 3 or np.any(x <= 0) or np.allclose(x, x[0]):
        return pd.Series(np.nan, index=values.index)
    shape, _, scale = stats.gamma.fit(x, floc=0)
    probability = np.clip(stats.gamma.cdf(x, shape, loc=0, scale=scale), 1e-6, 1 - 1e-6)
    return pd.Series(stats.norm.ppf(probability), index=values.index)


def prepare_variables(panel: pd.DataFrame) -> pd.DataFrame:
    data = panel.sort_values(["huc12", "year"]).copy()
    data["log_irrigation"] = np.log(data["irrigation_mgd"])
    data["log_precipitation"] = np.log(data["effective_precipitation_mgd"])
    data["spi_proxy"] = data.groupby("huc12", group_keys=False)["effective_precipitation_mgd"].apply(
        standardized_precipitation_proxy
    )
    data["unit_precip_percentile"] = data.groupby("huc12")["effective_precipitation_mgd"].rank(pct=True)
    data["lagged_irrigation_mgd"] = data.groupby("huc12")["irrigation_mgd"].shift(1)
    data["lagged_log_irrigation"] = np.log(data["lagged_irrigation_mgd"])
    return data


def _absorb_fixed_effects(data: pd.DataFrame, columns: list[str], effects: list[str]) -> np.ndarray:
    """Remove multiple fixed effects by alternating projections."""

    residual = data[columns].astype(float).copy()
    for _ in range(200):
        previous = residual.to_numpy(copy=True)
        for effect in effects:
            residual -= residual.groupby(data[effect]).transform("mean")
        if np.max(np.abs(residual.to_numpy() - previous)) < 1e-11:
            break
    else:
        raise RuntimeError("fixed-effect absorption did not converge")
    return residual.to_numpy()


def fixed_effect_regression(
    data: pd.DataFrame,
    outcome: str,
    regressors: list[str],
    *,
    specification: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """OLS after unit/year absorption with HUC12-clustered CR1 inference."""

    required = [outcome, *regressors, "huc12", "year"]
    sample = data.dropna(subset=required).copy()
    transformed = _absorb_fixed_effects(sample, [outcome, *regressors], ["huc12", "year"])
    y = transformed[:, 0]
    x = transformed[:, 1:]
    beta, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    residual = y - x @ beta
    bread = np.linalg.inv(x.T @ x)
    scores = x * residual[:, None]
    groups = sample["huc12"].to_numpy()
    unique_groups = pd.unique(groups)
    meat = np.zeros((len(regressors), len(regressors)))
    for group in unique_groups:
        score = scores[groups == group].sum(axis=0)
        meat += np.outer(score, score)
    n = len(sample)
    g = len(unique_groups)
    absorbed_rank = sample["huc12"].nunique() + sample["year"].nunique() - 1
    residual_df = n - absorbed_rank - len(regressors)
    correction = (g / (g - 1)) * ((n - 1) / residual_df)
    covariance = correction * bread @ meat @ bread
    standard_error = np.sqrt(np.diag(covariance))
    t_value = beta / standard_error
    p_value = 2 * stats.t.sf(np.abs(t_value), df=g - 1)
    critical = stats.t.ppf(0.975, df=g - 1)
    rows = []
    for index, term in enumerate(regressors):
        rows.append(
            {
                "specification": specification,
                "term": term,
                "estimate": beta[index],
                "clustered_se": standard_error[index],
                "ci_95_low": beta[index] - critical * standard_error[index],
                "ci_95_high": beta[index] + critical * standard_error[index],
                "p_value": p_value[index],
                "observations": n,
                "huc12_clusters": g,
                "start_year": int(sample["year"].min()),
                "end_year": int(sample["year"].max()),
            }
        )
    return pd.DataFrame(rows), residual


def elasticity_model(data: pd.DataFrame, specification: str) -> pd.DataFrame:
    result, _ = fixed_effect_regression(
        data,
        "log_irrigation",
        ["log_precipitation", "temperature_f"],
        specification=specification,
    )
    return result


def drought_response_model(
    data: pd.DataFrame,
    *,
    specification: str,
    cutoff: float | None = PRIMARY_DROUGHT_CUTOFF,
    percentile_definition: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample = data.copy()
    if percentile_definition:
        sample["drought"] = sample["unit_precip_percentile"].le(0.20).astype(int)
        definition = "within-unit effective precipitation <= 20th percentile"
    else:
        sample["drought"] = sample["spi_proxy"].le(float(cutoff)).astype(int)
        definition = f"SPI proxy <= {cutoff:g}"
    center = sample["log_precipitation"].mean()
    sample["log_precipitation_centered"] = sample["log_precipitation"] - center
    sample["precipitation_x_drought"] = sample["log_precipitation_centered"] * sample["drought"]
    coefficients, _ = fixed_effect_regression(
        sample,
        "log_irrigation",
        ["log_precipitation_centered", "drought", "precipitation_x_drought", "temperature_f"],
        specification=specification,
    )
    indexed = coefficients.set_index("term")
    non_drought = indexed.loc["log_precipitation_centered"]
    interaction = indexed.loc["precipitation_x_drought"]

    # Refit transformed design only to recover covariance for the linear sum.
    clean = sample.dropna(
        subset=[
            "log_irrigation",
            "log_precipitation_centered",
            "drought",
            "precipitation_x_drought",
            "temperature_f",
        ]
    ).copy()
    transformed = _absorb_fixed_effects(
        clean,
        [
            "log_irrigation",
            "log_precipitation_centered",
            "drought",
            "precipitation_x_drought",
            "temperature_f",
        ],
        ["huc12", "year"],
    )
    y, x = transformed[:, 0], transformed[:, 1:]
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ beta
    bread = np.linalg.inv(x.T @ x)
    groups = clean["huc12"].to_numpy()
    unique_groups = pd.unique(groups)
    scores = x * residual[:, None]
    meat = np.zeros((x.shape[1], x.shape[1]))
    for group in unique_groups:
        score = scores[groups == group].sum(axis=0)
        meat += np.outer(score, score)
    n, g = len(clean), len(unique_groups)
    absorbed_rank = clean["huc12"].nunique() + clean["year"].nunique() - 1
    correction = (g / (g - 1)) * ((n - 1) / (n - absorbed_rank - x.shape[1]))
    covariance = correction * bread @ meat @ bread
    contrast = np.array([1.0, 0.0, 1.0, 0.0])
    drought_elasticity = float(contrast @ beta)
    drought_se = float(np.sqrt(contrast @ covariance @ contrast))
    critical = stats.t.ppf(0.975, df=g - 1)
    drought_p = 2 * stats.t.sf(abs(drought_elasticity / drought_se), df=g - 1)

    summary = pd.DataFrame(
        [
            {
                "specification": specification,
                "drought_definition": definition,
                "non_drought_elasticity": non_drought["estimate"],
                "non_drought_clustered_se": non_drought["clustered_se"],
                "non_drought_ci_95_low": non_drought["ci_95_low"],
                "non_drought_ci_95_high": non_drought["ci_95_high"],
                "drought_elasticity": drought_elasticity,
                "drought_clustered_se": drought_se,
                "drought_ci_95_low": drought_elasticity - critical * drought_se,
                "drought_ci_95_high": drought_elasticity + critical * drought_se,
                "drought_elasticity_p_value": drought_p,
                "elasticity_difference": interaction["estimate"],
                "difference_clustered_se": interaction["clustered_se"],
                "difference_ci_95_low": interaction["ci_95_low"],
                "difference_ci_95_high": interaction["ci_95_high"],
                "difference_p_value": interaction["p_value"],
                "drought_observations": int(sample["drought"].sum()),
                "non_drought_observations": int(sample["drought"].eq(0).sum()),
                "observations": int(non_drought["observations"]),
                "huc12_clusters": int(non_drought["huc12_clusters"]),
                "start_year": int(sample["year"].min()),
                "end_year": int(sample["year"].max()),
            }
        ]
    )
    return summary, coefficients


def drought_descriptives(data: pd.DataFrame, cutoff: float = PRIMARY_DROUGHT_CUTOFF) -> pd.DataFrame:
    sample = data.copy()
    sample["drought"] = sample["spi_proxy"].le(cutoff)
    sample["irrigation_relative_to_unit_mean"] = sample["irrigation_mgd"] / sample.groupby("huc12")[
        "irrigation_mgd"
    ].transform("mean")
    return (
        sample.groupby("drought", as_index=False)
        .agg(
            observations=("irrigation_mgd", "size"),
            mean_irrigation_mgd=("irrigation_mgd", "mean"),
            median_irrigation_mgd=("irrigation_mgd", "median"),
            mean_within_unit_irrigation_index=("irrigation_relative_to_unit_mean", "mean"),
            mean_effective_precipitation_mgd=("effective_precipitation_mgd", "mean"),
            mean_spi_proxy=("spi_proxy", "mean"),
        )
        .assign(drought_definition=f"SPI proxy <= {cutoff:g}")
    )


def validation(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_data = data.dropna(subset=["lagged_log_irrigation"]).copy()
    train = model_data[model_data["year"].between(2001, 2013)].copy()
    test = model_data[model_data["year"].between(2014, 2020)].copy()
    numeric = ["log_precipitation", "temperature_f", "lagged_log_irrigation"]
    features = [*numeric, "huc12"]
    preprocess = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric),
            ("unit", OneHotEncoder(handle_unknown="ignore"), ["huc12"]),
        ]
    )
    model = make_pipeline(preprocess, Ridge(alpha=1.0, solver="lsqr"))
    model.fit(train[features], train["log_irrigation"])
    train_log_prediction = model.predict(train[features])
    smearing = float(np.mean(np.exp(train["log_irrigation"] - train_log_prediction)))
    test["model_prediction_mgd"] = np.exp(model.predict(test[features])) * smearing
    test["naive_prediction_mgd"] = test["lagged_irrigation_mgd"]
    metrics = []
    for name, prediction in [
        ("precip_temperature_lag_ridge", test["model_prediction_mgd"]),
        ("naive_last_year", test["naive_prediction_mgd"]),
    ]:
        metrics.append(
            {
                "model": name,
                "train_window": "2001-2013 (2000 supplies first lag)",
                "test_window": "2014-2020",
                "rmse_mgd": mean_squared_error(test["irrigation_mgd"], prediction) ** 0.5,
                "mae_mgd": mean_absolute_error(test["irrigation_mgd"], prediction),
                "log_rmse": mean_squared_error(np.log(test["irrigation_mgd"]), np.log(prediction)) ** 0.5,
                "test_observations": len(test),
                "huc12_units": test["huc12"].nunique(),
                "smearing_factor": smearing if name.startswith("precip") else np.nan,
            }
        )
    predictions = test[
        ["huc12", "huc4", "year", "irrigation_mgd", "model_prediction_mgd", "naive_prediction_mgd"]
    ]
    return pd.DataFrame(metrics), predictions


def robustness_grid(primary: pd.DataFrame, full_panel: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(data: pd.DataFrame, name: str, cutoff: float | None = -1.0, percentile: bool = False) -> None:
        elasticity = elasticity_model(data, name).query("term == 'log_precipitation'").iloc[0]
        drought, _ = drought_response_model(
            data, specification=name, cutoff=cutoff, percentile_definition=percentile
        )
        row = drought.iloc[0].to_dict()
        row.update(
            {
                "overall_elasticity": elasticity["estimate"],
                "overall_clustered_se": elasticity["clustered_se"],
                "overall_ci_95_low": elasticity["ci_95_low"],
                "overall_ci_95_high": elasticity["ci_95_high"],
                "overall_p_value": elasticity["p_value"],
            }
        )
        rows.append(row)

    add(primary, "primary_spi_le_minus_1", cutoff=-1.0)
    add(primary, "drought_spi_le_minus_0_8", cutoff=-0.8)
    add(primary, "drought_spi_le_minus_1_5", cutoff=-1.5)
    add(primary, "drought_bottom_20_percent", cutoff=None, percentile=True)

    means = primary.groupby("huc12")["irrigation_mgd"].mean()
    cutoff_value = means.quantile(0.99)
    no_outliers = primary[~primary["huc12"].isin(means[means > cutoff_value].index)]
    add(no_outliers, "drop_top_1_percent_mean_irrigation", cutoff=-1.0)

    add(select_balanced_positive(full_panel, 2000, 2017), "window_2000_2017", cutoff=-1.0)
    add(select_balanced_positive(full_panel, 2005, 2020), "window_2005_2020", cutoff=-1.0)
    add(select_unbalanced(full_panel, 15), "unbalanced_at_least_15_years", cutoff=-1.0)
    result = pd.DataFrame(rows)
    result["outlier_cutoff_mean_irrigation_mgd"] = np.where(
        result["specification"].eq("drop_top_1_percent_mean_irrigation"), cutoff_value, np.nan
    )
    return result


def _spec_text(readme: Path) -> str:
    text = readme.read_text()
    return text.split("## Pre-specified irrigation-response analysis", 1)[1].split(
        "<!-- IRRIGATION_RESULTS_START -->", 1
    )[0]


def verify_specification(readme: Path) -> None:
    digest = hashlib.sha256(_spec_text(readme).encode()).hexdigest()
    if digest != SPEC_SHA256:
        raise RuntimeError(
            "the pre-specified README section changed; review it and intentionally update SPEC_SHA256 before running"
        )


def make_charts(
    drought: pd.DataFrame,
    robust: pd.DataFrame,
    predictions: pd.DataFrame,
    results_dir: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    row = drought.iloc[0]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    estimates = [row["non_drought_elasticity"], row["drought_elasticity"]]
    lows = [row["non_drought_ci_95_low"], row["drought_ci_95_low"]]
    highs = [row["non_drought_ci_95_high"], row["drought_ci_95_high"]]
    ax.errorbar(
        ["Non-drought", "Drought"],
        estimates,
        yerr=[np.array(estimates) - np.array(lows), np.array(highs) - np.array(estimates)],
        fmt="o",
        capsize=5,
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Precipitation elasticity of irrigation use")
    ax.set_title("Elasticity by drought status (95% clustered CI)")
    fig.tight_layout()
    fig.savefig(results_dir / "irrigation_drought_elasticity.png", dpi=180)
    plt.close(fig)

    plot = robust.sort_values("overall_elasticity")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.errorbar(
        plot["overall_elasticity"],
        plot["specification"],
        xerr=[
            plot["overall_elasticity"] - plot["overall_ci_95_low"],
            plot["overall_ci_95_high"] - plot["overall_elasticity"],
        ],
        fmt="o",
        capsize=3,
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Overall precipitation elasticity (95% clustered CI)")
    ax.set_title("Pre-specified robustness specifications")
    fig.tight_layout()
    fig.savefig(results_dir / "irrigation_elasticity_robustness.png", dpi=180)
    plt.close(fig)

    yearly = predictions.groupby("year", as_index=False).agg(
        actual_mgd=("irrigation_mgd", "mean"),
        model_prediction_mgd=("model_prediction_mgd", "mean"),
        naive_prediction_mgd=("naive_prediction_mgd", "mean"),
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    for column, label in [
        ("actual_mgd", "Actual"),
        ("model_prediction_mgd", "Pre-specified ridge"),
        ("naive_prediction_mgd", "Last-year baseline"),
    ]:
        ax.plot(yearly["year"], yearly[column], marker="o", label=label)
    ax.set_ylabel("Mean HUC12 irrigation consumptive use (MGD)")
    ax.set_title("Out-of-time validation: 2014–2020")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "irrigation_out_of_time_validation.png", dpi=180)
    plt.close(fig)


def update_readme(
    readme: Path,
    availability: pd.DataFrame,
    elasticity: pd.DataFrame,
    drought: pd.DataFrame,
    validation_metrics: pd.DataFrame,
    robust: pd.DataFrame,
    identity: pd.DataFrame,
    drought_descriptive: pd.DataFrame,
) -> None:
    text = readme.read_text()
    start = "<!-- IRRIGATION_RESULTS_START -->"
    end = "<!-- IRRIGATION_RESULTS_END -->"
    a = availability.iloc[0]
    e = elasticity.query("term == 'log_precipitation'").iloc[0]
    d = drought.iloc[0]
    metrics = validation_metrics.set_index("model")
    model = metrics.loc["precip_temperature_lag_ridge"]
    naive = metrics.loc["naive_last_year"]
    robust_min = robust.loc[robust["overall_elasticity"].idxmin()]
    robust_max = robust.loc[robust["overall_elasticity"].idxmax()]
    source_check = identity.iloc[0]
    descriptive = drought_descriptive.set_index("drought")
    non_drought_desc = descriptive.loc[False]
    drought_desc = descriptive.loc[True]
    generated = f"""{start}
## Irrigation-response results

The source audit found **{int(a.huc12_columns_in_nine_huc4_regions):,} HUC12 columns** in the nine California HUC4 regions. The primary balanced-positive sample contains **{int(e.huc12_clusters):,} HUC12 units and {int(e.observations):,} unit-year observations** from {int(e.start_year)}–{int(e.end_year)}. The source-identity check covers **{int(source_check.matched_unit_years):,} matched unit-years** and reproduces annualized monthly irrigation as `Irrigation_wateruse − PPTeff` with maximum absolute discrepancy **{source_check.max_absolute_identity_error_mgd:.6f} MGD**, confirming that the ambiguous annual file is total crop ET rather than irrigation consumptive use.

The estimated precipitation elasticity is **{e.estimate:.3f}** (HUC12-clustered SE **{e.clustered_se:.3f}**, 95% CI **{e.ci_95_low:.3f} to {e.ci_95_high:.3f}**, p = **{e.p_value:.3g}**). Thus, within this modeled panel, a 1% increase in effective precipitation is associated with a **{abs(e.estimate):.3f}% {'decrease' if e.estimate < 0 else 'increase'}** in irrigation consumptive use, conditional on HUC12 and year fixed effects and HUC4 temperature.

Under the pre-specified SPI-proxy ≤ −1.0 definition, there are **{int(d.drought_observations):,} drought observations** and **{int(d.non_drought_observations):,} non-drought observations**. Raw mean irrigation is **{drought_desc.mean_irrigation_mgd:.3f} MGD** in drought observations versus **{non_drought_desc.mean_irrigation_mgd:.3f} MGD** otherwise, but this composition-sensitive comparison reverses after scaling each observation by its own unit mean: **{drought_desc.mean_within_unit_irrigation_index:.3f}** in drought versus **{non_drought_desc.mean_within_unit_irrigation_index:.3f}** otherwise. Estimated elasticity is **{d.non_drought_elasticity:.3f}** outside drought (95% CI **{d.non_drought_ci_95_low:.3f} to {d.non_drought_ci_95_high:.3f}**) and **{d.drought_elasticity:.3f}** during drought (95% CI **{d.drought_ci_95_low:.3f} to {d.drought_ci_95_high:.3f}**). The drought-minus-non-drought elasticity difference is **{d.elasticity_difference:.3f}** (95% CI **{d.difference_ci_95_low:.3f} to {d.difference_ci_95_high:.3f}**, p = **{d.difference_p_value:.3g}**).

For 2014–2020 out-of-time validation, the pre-specified ridge model has RMSE **{model.rmse_mgd:.3f} MGD** and MAE **{model.mae_mgd:.3f} MGD**. The naive last-year baseline has RMSE **{naive.rmse_mgd:.3f} MGD** and MAE **{naive.mae_mgd:.3f} MGD**. The model {'beats' if model.rmse_mgd < naive.rmse_mgd and model.mae_mgd < naive.mae_mgd else 'does not beat'} the naive baseline on both level-error metrics.

Across the full pre-specified robustness grid, the overall elasticity ranges from **{robust_min.overall_elasticity:.3f}** (`{robust_min.specification}`) to **{robust_max.overall_elasticity:.3f}** (`{robust_max.specification}`). Every requested specification is retained in `results/irrigation_robustness.csv` regardless of significance.

These estimates are descriptive associations within USGS model output. The positive sign does **not** establish that additional rainfall causes farmers to irrigate more: both total quantities can rise with time-varying irrigated extent, and effective precipitation and irrigation consumptive use are jointly separated by the same soil-water-balance procedure. The HUC12 selection follows the nine hydrologic regions and is not a county-boundary or exact California-state-boundary clip. These results must not be presented as causal behavioral elasticities.
{end}"""
    before, remainder = text.split(start, 1)
    _, after = remainder.split(end, 1)
    readme.write_text(before + generated + after)


def run(data_dir: Path, results_dir: Path, readme: Path) -> None:
    verify_specification(readme)
    results_dir.mkdir(parents=True, exist_ok=True)
    full_panel, availability = build_huc12_panel(data_dir)
    annualized_irrigation = full_panel[["huc12", "year", "irrigation_mgd"]].copy()
    identity = source_identity_check(data_dir, annualized_irrigation)
    primary = select_balanced_positive(full_panel)
    elasticity = elasticity_model(primary, "primary_2000_2020_balanced_positive")
    drought, drought_coefficients = drought_response_model(
        primary, specification="primary_spi_le_minus_1", cutoff=-1.0
    )
    drought_summary = drought_descriptives(primary)
    metrics, predictions = validation(primary)
    robust = robustness_grid(primary, full_panel)

    outputs = {
        "irrigation_data_availability.csv": availability,
        "irrigation_source_identity_check.csv": identity,
        "irrigation_huc12_panel.csv": primary,
        "irrigation_elasticity.csv": elasticity,
        "irrigation_drought_response.csv": drought,
        "irrigation_drought_coefficients.csv": drought_coefficients,
        "irrigation_drought_descriptives.csv": drought_summary,
        "irrigation_validation_metrics.csv": metrics,
        "irrigation_validation_predictions.csv": predictions,
        "irrigation_robustness.csv": robust,
    }
    for filename, frame in outputs.items():
        frame.to_csv(results_dir / filename, index=False)
    config = {
        "specification_sha256": SPEC_SHA256,
        "primary_years": "2000-2020",
        "primary_drought_cutoff": -1.0,
        "validation_train_years": "2001-2013",
        "validation_test_years": "2014-2020",
        "ridge_alpha": 1.0,
        "robustness_specifications": robust["specification"].tolist(),
    }
    (results_dir / "irrigation_run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    make_charts(drought, robust, predictions, results_dir)
    update_readme(readme, availability, elasticity, drought, metrics, robust, identity, drought_summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path.cwd())
    parser.add_argument("--results-dir", type=Path, default=Path.cwd() / "results")
    parser.add_argument("--readme", type=Path, default=Path.cwd() / "README.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.data_dir, args.results_dir, args.readme)


if __name__ == "__main__":
    main()
