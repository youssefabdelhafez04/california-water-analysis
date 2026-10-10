"""End-to-end policy analysis and reproducible result generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .analysis import AnalysisWindow, assign_treatment, bootstrap_median_ci, compare_windows
from .data import build_annual_panel


def _fit_clustered(formula: str, data: pd.DataFrame):
    return smf.ols(formula, data=data).fit(
        cov_type="cluster",
        cov_kwds={"groups": data["watershed"], "use_correction": True},
    )


def total_panel(panel: pd.DataFrame) -> pd.DataFrame:
    totals = (
        panel.groupby(["watershed", "huc4", "year"], as_index=False)
        .agg(
            total_use_mgd=("annual_avg_use_mgd", "sum"),
            precipitation_in=("precipitation_in", "first"),
            temperature_f=("temperature_f", "first"),
        )
    )
    return totals


def add_rainfall_adjustment(
    totals: pd.DataFrame,
    comparison: pd.DataFrame,
    window: AnalysisWindow,
) -> pd.DataFrame:
    rows = []
    for watershed, group in totals.groupby("watershed"):
        pre_fit = group[group["year"] < window.post_start]
        model = smf.ols("total_use_mgd ~ precipitation_in", data=pre_fit).fit()
        post = group[group["year"].between(window.post_start, window.post_end)].copy()
        post["counterfactual_mgd"] = model.predict(post)
        actual = post["total_use_mgd"].mean()
        counterfactual = post["counterfactual_mgd"].mean()
        rows.append(
            {
                "watershed": watershed,
                "rain_adjusted_change_mgd": actual - counterfactual,
                "rain_adjusted_percent_change": 100 * (actual - counterfactual) / counterfactual,
                "pre_2018_rain_slope_mgd_per_in": model.params["precipitation_in"],
                "pre_2018_r_squared": model.rsquared,
            }
        )
    return comparison.merge(pd.DataFrame(rows), on="watershed", validate="one_to_one")


def summarize_changes(changes: pd.DataFrame, window: AnalysisWindow) -> pd.DataFrame:
    raw_lo, raw_hi = bootstrap_median_ci(changes["percent_change"])
    adj_lo, adj_hi = bootstrap_median_ci(changes["rain_adjusted_percent_change"])
    largest_drop = changes.loc[changes["percent_change"].idxmin()]
    largest_increase = changes.loc[changes["percent_change"].idxmax()]
    metrics = {
        "pre_window": f"{window.pre_start}-{window.pre_end}",
        "post_window": f"{window.post_start}-{window.post_end}",
        "watershed_count": len(changes),
        "median_raw_percent_change": changes["percent_change"].median(),
        "median_raw_percent_change_ci_low": raw_lo,
        "median_raw_percent_change_ci_high": raw_hi,
        "median_rain_adjusted_percent_change": changes["rain_adjusted_percent_change"].median(),
        "median_rain_adjusted_percent_change_ci_low": adj_lo,
        "median_rain_adjusted_percent_change_ci_high": adj_hi,
        "watersheds_decreased": int(changes["absolute_change_mgd"].lt(0).sum()),
        "watersheds_increased": int(changes["absolute_change_mgd"].gt(0).sum()),
        "watersheds_unchanged": int(changes["absolute_change_mgd"].eq(0).sum()),
        "largest_raw_decrease_watershed": largest_drop["watershed"],
        "largest_raw_decrease_percent": largest_drop["percent_change"],
        "largest_raw_increase_watershed": largest_increase["watershed"],
        "largest_raw_increase_percent": largest_increase["percent_change"],
    }
    return pd.DataFrame([metrics])


def did_data(panel: pd.DataFrame) -> pd.DataFrame:
    data = panel.copy()
    data["treated"] = data["category"].map(assign_treatment)
    data["unit"] = data["watershed"] + "__" + data["category"]
    return data


def fit_did(panel: pd.DataFrame, policy_year: int = 2018) -> tuple[object, pd.DataFrame]:
    data = did_data(panel)
    data["post"] = data["year"].ge(policy_year).astype(int)
    data["did"] = data["treated"] * data["post"]
    model = _fit_clustered(
        "annual_avg_use_mgd ~ did + precipitation_in + temperature_f + C(unit) + C(year)", data
    )
    ci = model.conf_int().loc["did"]
    result = pd.DataFrame(
        [
            {
                "specification": "category_DiD_unit_and_year_FE",
                "policy_year": policy_year,
                "effect_mgd": model.params["did"],
                "clustered_se_mgd": model.bse["did"],
                "ci_95_low_mgd": ci.iloc[0],
                "ci_95_high_mgd": ci.iloc[1],
                "p_value": model.pvalues["did"],
                "watershed_clusters": data["watershed"].nunique(),
                "observations": int(model.nobs),
            }
        ]
    )
    return model, result


def fit_event_study(panel: pd.DataFrame, reference_year: int = 2017) -> tuple[object, pd.DataFrame, float]:
    data = did_data(panel)
    event_terms = []
    year_to_term = {}
    for year in sorted(data["year"].unique()):
        if year == reference_year:
            continue
        suffix = f"m{reference_year - year}" if year < reference_year else f"p{year - reference_year}"
        term = f"event_{suffix}"
        data[term] = data["treated"] * data["year"].eq(year).astype(int)
        event_terms.append(term)
        year_to_term[year] = term
    formula = (
        "annual_avg_use_mgd ~ "
        + " + ".join(event_terms)
        + " + precipitation_in + temperature_f + C(unit) + C(year)"
    )
    model = _fit_clustered(formula, data)
    rows = [{"year": reference_year, "relative_year": 0, "effect_mgd": 0.0, "se_mgd": 0.0, "ci_95_low_mgd": 0.0, "ci_95_high_mgd": 0.0}]
    for year, term in year_to_term.items():
        effect = model.params[term]
        se = model.bse[term]
        rows.append(
            {
                "year": year,
                "relative_year": year - reference_year,
                "effect_mgd": effect,
                "se_mgd": se,
                "ci_95_low_mgd": effect - 1.96 * se,
                "ci_95_high_mgd": effect + 1.96 * se,
            }
        )
    # With only nine watershed clusters, an all-preperiod joint restriction is
    # rank deficient. Test the seven recent pre-policy coefficients instead;
    # this keeps the restriction rank below the number of clusters.
    pre_terms = [term for year, term in year_to_term.items() if 2010 <= year < reference_year]
    restriction = ", ".join(f"{term} = 0" for term in pre_terms)
    pretrend_p = float(model.wald_test(restriction, scalar=True).pvalue)
    events = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    events["joint_2010_2016_pretrend_p_value"] = pretrend_p
    return model, events, pretrend_p


def placebo_test(panel: pd.DataFrame) -> pd.DataFrame:
    pre_policy = panel[panel["year"] <= 2017].copy()
    _, result = fit_did(pre_policy, policy_year=2015)
    result["specification"] = "placebo_2015_using_2000_2017_only"
    return result


def robustness(panel: pd.DataFrame) -> pd.DataFrame:
    totals = total_panel(panel)
    statewide_weather = totals.groupby("year")["precipitation_in"].mean()
    drought_threshold = statewide_weather.quantile(0.10)
    extreme_drought_years = statewide_weather[statewide_weather <= drought_threshold].index.tolist()
    rows = []
    specs = [
        ("2010-2020", 2010, 2020, False),
        ("2012-2020", 2012, 2020, False),
        ("2014-2020", 2014, 2020, False),
        ("2000-2020_drop_bottom_precip_decile", 2000, 2020, True),
    ]
    for name, start, end, drop_drought in specs:
        subset = panel[panel["year"].between(start, end)].copy()
        dropped = []
        if drop_drought:
            dropped = extreme_drought_years
            subset = subset[~subset["year"].isin(dropped)]
        _, did = fit_did(subset)
        did_row = did.iloc[0]
        rows.append(
            {
                "specification": name,
                "start_year": start,
                "end_year": end,
                "dropped_years": ";".join(map(str, dropped)),
                "did_effect_mgd": did_row["effect_mgd"],
                "did_clustered_se_mgd": did_row["clustered_se_mgd"],
                "did_ci_95_low_mgd": did_row["ci_95_low_mgd"],
                "did_ci_95_high_mgd": did_row["ci_95_high_mgd"],
                "did_p_value": did_row["p_value"],
            }
        )
    for start in [2012, 2014, 2015]:
        window = AnalysisWindow(start, 2017, 2018, 2020)
        change = compare_windows(totals, window)
        rows.append(
            {
                "specification": f"raw_before_after_{start}_2017_vs_2018_2020",
                "start_year": start,
                "end_year": 2020,
                "dropped_years": "",
                "median_raw_percent_change": change["percent_change"].median(),
            }
        )
    return pd.DataFrame(rows)


def prediction_baseline(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = panel[panel["category"].eq("irrigation")].copy()
    data = data.sort_values(["watershed", "year"])
    data["lagged_use_mgd"] = data.groupby("watershed")["annual_avg_use_mgd"].shift(1)
    data = data.dropna(subset=["lagged_use_mgd", "precipitation_in", "temperature_f"])
    train = data[data["year"].between(2000, 2013)].copy()
    test = data[data["year"].between(2014, 2017)].copy()
    features = ["precipitation_in", "temperature_f", "lagged_use_mgd", "watershed"]
    preprocess = ColumnTransformer(
        [
            ("numeric", StandardScaler(), features[:3]),
            ("watershed", OneHotEncoder(handle_unknown="ignore", drop="first"), ["watershed"]),
        ]
    )
    model = make_pipeline(preprocess, RidgeCV(alphas=np.logspace(-4, 4, 81)))
    model.fit(train[features], train["annual_avg_use_mgd"])
    test["model_prediction_mgd"] = model.predict(test[features])
    test["naive_prediction_mgd"] = test["lagged_use_mgd"]
    actual = test["annual_avg_use_mgd"]
    metrics = []
    for name, prediction in [
        ("weather_lag_ridge", test["model_prediction_mgd"]),
        ("naive_last_year", test["naive_prediction_mgd"]),
    ]:
        metrics.append(
            {
                "model": name,
                "train_window": "2000-2013",
                "test_window": "2014-2017",
                "rmse_mgd": mean_squared_error(actual, prediction) ** 0.5,
                "mae_mgd": mean_absolute_error(actual, prediction),
                "test_observations": len(test),
            }
        )
    predictions = test[
        ["watershed", "year", "annual_avg_use_mgd", "model_prediction_mgd", "naive_prediction_mgd"]
    ].rename(columns={"annual_avg_use_mgd": "actual_mgd"})
    return pd.DataFrame(metrics), predictions


def make_charts(
    panel: pd.DataFrame,
    changes: pd.DataFrame,
    events: pd.DataFrame,
    predictions: pd.DataFrame,
    results_dir: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    ordered = changes.sort_values("percent_change")
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = np.where(ordered["percent_change"] < 0, "#1f77b4", "#d95f02")
    ax.barh(ordered["watershed"], ordered["percent_change"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Raw change in average annual water use (%)")
    ax.set_title("Water use after 2018 versus 2014–2017")
    fig.tight_layout()
    fig.savefig(results_dir / "watershed_before_after.png", dpi=180)
    plt.close(fig)

    trends = panel.groupby(["category", "year"], as_index=False)["annual_avg_use_mgd"].mean()
    base = trends[trends["year"].eq(2017)].set_index("category")["annual_avg_use_mgd"]
    trends["index_2017_100"] = trends.apply(lambda r: 100 * r["annual_avg_use_mgd"] / base[r["category"]], axis=1)
    fig, ax = plt.subplots(figsize=(9, 5))
    for category, group in trends.groupby("category"):
        ax.plot(group["year"], group["index_2017_100"], marker="o", label=category.replace("_", " ").title())
    ax.axvline(2018, color="black", linestyle="--", linewidth=1, label="2018 policy")
    ax.set_ylabel("Mean watershed use (2017 = 100)")
    ax.set_title("Category trends before and after the 2018 legislation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "parallel_trends.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(
        events["year"],
        events["effect_mgd"],
        yerr=[events["effect_mgd"] - events["ci_95_low_mgd"], events["ci_95_high_mgd"] - events["effect_mgd"]],
        fmt="o-",
        capsize=3,
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(2018, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("Public-supply differential relative to 2017 (MGD)")
    ax.set_title("Event study with watershed-category and year fixed effects")
    fig.tight_layout()
    fig.savefig(results_dir / "event_study.png", dpi=180)
    plt.close(fig)

    yearly = predictions.groupby("year", as_index=False).agg(
        actual_mgd=("actual_mgd", "mean"),
        model_prediction_mgd=("model_prediction_mgd", "mean"),
        naive_prediction_mgd=("naive_prediction_mgd", "mean"),
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    for column, label in [
        ("actual_mgd", "Actual"),
        ("model_prediction_mgd", "Weather + lag ridge"),
        ("naive_prediction_mgd", "Last-year baseline"),
    ]:
        ax.plot(yearly["year"], yearly[column], marker="o", label=label)
    ax.set_ylabel("Mean watershed irrigation use (MGD)")
    ax.set_title("Pre-policy prediction holdout: 2014–2017")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "prediction_baseline.png", dpi=180)
    plt.close(fig)


def update_readme(
    readme: Path,
    summary: pd.DataFrame,
    did: pd.DataFrame,
    events: pd.DataFrame,
    placebo: pd.DataFrame,
    metrics: pd.DataFrame,
    robustness_table: pd.DataFrame,
) -> None:
    text = readme.read_text()
    start = "<!-- GENERATED_RESULTS_START -->"
    end = "<!-- GENERATED_RESULTS_END -->"
    s = summary.iloc[0]
    d = did.iloc[0]
    p = placebo.iloc[0]
    model = metrics.set_index("model")
    ridge = model.loc["weather_lag_ridge"]
    naive = model.loc["naive_last_year"]
    drought = robustness_table[robustness_table["specification"].eq("2000-2020_drop_bottom_precip_decile")].iloc[0]
    placebo_conclusion = (
        "The placebo is statistically distinguishable from zero at the 5% level, so this falsification check fails."
        if p.p_value < 0.05
        else "The placebo is not statistically distinguishable from zero at the 5% level."
    )
    prediction_conclusion = (
        "The ridge model does not unambiguously beat the naive baseline across both error metrics."
        if (ridge.rmse_mgd < naive.rmse_mgd) != (ridge.mae_mgd < naive.mae_mgd)
        else "The ridge model beats the naive baseline on both reported error metrics."
        if ridge.rmse_mgd < naive.rmse_mgd
        else "The ridge model does not beat the naive baseline on either reported error metric."
    )
    generated = f"""{start}
## Results from the reproducible run

The default {s.pre_window} versus {s.post_window} comparison covers {int(s.watershed_count)} watersheds. Median raw use changed by **{s.median_raw_percent_change:.2f}%** (watershed bootstrap 95% CI **{s.median_raw_percent_change_ci_low:.2f}% to {s.median_raw_percent_change_ci_high:.2f}%**); {int(s.watersheds_decreased)} watersheds decreased and {int(s.watersheds_increased)} increased. The largest decrease was **{s.largest_raw_decrease_watershed} ({s.largest_raw_decrease_percent:.2f}%)**, while the largest increase was **{s.largest_raw_increase_watershed} ({s.largest_raw_increase_percent:.2f}%)**. After projecting each watershed's post-2018 precipitation through its own pre-2018 use-on-rainfall regression, the median post-policy anomaly was **{s.median_rain_adjusted_percent_change:.2f}%** (95% bootstrap CI **{s.median_rain_adjusted_percent_change_ci_low:.2f}% to {s.median_rain_adjusted_percent_change_ci_high:.2f}%**).

The exploratory category DiD estimate is **{d.effect_mgd:.2f} MGD** (watershed-clustered SE **{d.clustered_se_mgd:.2f}**, 95% CI **{d.ci_95_low_mgd:.2f} to {d.ci_95_high_mgd:.2f}**, p = **{d.p_value:.3g}**; {int(d.watershed_clusters)} clusters). The rank-valid joint test of the 2010–2016 event-study pre-period coefficients has p = **{events.joint_2010_2016_pretrend_p_value.iloc[0]:.3g}**, contradicting parallel trends at conventional levels. The 2015 placebo estimate is **{p.effect_mgd:.2f} MGD** (95% CI **{p.ci_95_low_mgd:.2f} to {p.ci_95_high_mgd:.2f}**, p = **{p.p_value:.3g}**). {placebo_conclusion} Dropping the data-defined bottom precipitation-decile years ({drought.dropped_years}) gives a DiD estimate of **{drought.did_effect_mgd:.2f} MGD** (95% CI **{drought.did_ci_95_low_mgd:.2f} to {drought.did_ci_95_high_mgd:.2f}**).

On the strictly pre-policy holdout (train 2000–2013; test 2014–2017), the precipitation/temperature/lagged-use ridge model has RMSE **{ridge.rmse_mgd:.2f} MGD** and MAE **{ridge.mae_mgd:.2f} MGD**. The naive last-year baseline has RMSE **{naive.rmse_mgd:.2f} MGD** and MAE **{naive.mae_mgd:.2f} MGD**. {prediction_conclusion}

These before/after, residual, and prediction results are associational. The DiD is also **not interpreted as causal**: both categories are statewide, irrigation is not a true untreated control, category-specific trends may differ, the pre-trend diagnostic is informative but not proof, and nine watershed clusters make cluster-robust inference fragile.
{end}"""
    before, remainder = text.split(start, 1)
    _, after = remainder.split(end, 1)
    readme.write_text(before + generated + after)


def run(data_dir: Path, results_dir: Path, window: AnalysisWindow, readme: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    panel = build_annual_panel(data_dir)
    totals = total_panel(panel)
    changes = compare_windows(totals, window)
    changes = add_rainfall_adjustment(totals, changes, window)
    summary = summarize_changes(changes, window)
    _, did = fit_did(panel)
    _, events, _ = fit_event_study(panel)
    placebo = placebo_test(panel)
    robust = robustness(panel)
    prediction_metrics, predictions = prediction_baseline(panel)

    outputs = {
        "annual_panel.csv": panel,
        "watershed_before_after.csv": changes,
        "before_after_summary.csv": summary,
        "did_results.csv": did,
        "event_study.csv": events,
        "placebo_2015.csv": placebo,
        "robustness.csv": robust,
        "prediction_metrics.csv": prediction_metrics,
        "prediction_predictions.csv": predictions,
    }
    for name, frame in outputs.items():
        frame.to_csv(results_dir / name, index=False)
    config = {
        "pre_start": window.pre_start,
        "pre_end": window.pre_end,
        "post_start": window.post_start,
        "post_end": window.post_end,
        "rainfall_fit_years": f"2000-{window.post_start - 1}",
        "bootstrap_resamples": 20_000,
        "random_seed": 2026,
    }
    (results_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    make_charts(panel, changes, events, predictions, results_dir)
    update_readme(readme, summary, did, events, placebo, prediction_metrics, robust)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path.cwd())
    parser.add_argument("--results-dir", type=Path, default=Path.cwd() / "results")
    parser.add_argument("--readme", type=Path, default=Path.cwd() / "README.md")
    parser.add_argument("--pre-start", type=int, default=2014)
    parser.add_argument("--pre-end", type=int, default=2017)
    parser.add_argument("--post-start", type=int, default=2018)
    parser.add_argument("--post-end", type=int, default=2020)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    window = AnalysisWindow(args.pre_start, args.pre_end, args.post_start, args.post_end)
    run(args.data_dir, args.results_dir, window, args.readme)


if __name__ == "__main__":
    main()
