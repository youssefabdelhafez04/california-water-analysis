# Methodology and statistical results

This document contains the detailed analysis behind the recruiter-facing [project README](../README.md). All values below come from the reproducible outputs in [`results/`](../results/); none are placeholders.

## Pre-specified irrigation-response analysis

The analysis was defined before the valid irrigation-response models were run. The script stores a SHA-256 digest of the specification in `results/irrigation_run_config.json`; the model grid was fixed rather than selected using significance.

A source-validation run initially treated the ambiguously named local `Irrigation_wateruse.csv` as irrigation use. An identity check showed that file is total crop evapotranspiration: subtracting effective precipitation reproduces the explicit monthly irrigation-consumptive-use series. That run was invalidated before reporting, and the amended specification fixed the outcome source before re-estimation.

### Question, data, and sample

The primary question is: **within a subwatershed, how strongly is modeled irrigation consumptive use associated with changes in effective precipitation, and does that association differ in drought years?**

The original files contain irrigation consumptive use and effective precipitation for 4,335 HUC12 columns located in the nine California HUC4 regions used by this project. The [USGS data release](https://water.usgs.gov/catalog/datasets/649711fe-637f-4cc0-9020-09a21e691d88/) covers all CONUS HUC12 subwatersheds from 2000–2020. The primary balanced panel contains 1,169 HUC12 units with positive irrigation and effective precipitation in all 21 years, or 24,549 unit-year observations. Values coded `999` are missing and are never analyzed. A pre-specified unbalanced robustness sample retains units with at least 15 positive paired years.

`Irr_CU_HUC12_Tot_monthly_2000_2020.csv` contains monthly mean crop-irrigation consumptive use in MGD and is annualized using calendar-day weights. `PPTeff_HUC12_Tot_annual_2000_2020.csv` contains annual effective precipitation over irrigated land, represented as an MGD-equivalent flow. `Irrigation_wateruse.csv` is excluded as an outcome because the source-identity check shows it contains total crop evapotranspiration. NOAA data provide three daily stations for each of nine HUC4 regions; annual HUC4 temperature averages are assigned to contained HUC12s as a coarse control.

USGS defines consumptive use as crop evapotranspiration supplied by irrigation rather than rain or snow. Both consumptive use and effective precipitation are separated inside the same modeled soil-water balance. The estimates are therefore modeled physical associations, not identified behavioral or causal effects.

### Primary elasticity model

The pre-specified log-log panel regression for 2000–2020 is:

```text
log(irrigation consumptive use MGD)
  = beta × log(effective precipitation MGD)
  + temperature
  + HUC12 fixed effects + year fixed effects + error
```

`beta` is the precipitation elasticity: the percent change in irrigation consumptive use associated with a 1% change in effective precipitation within the same HUC12, conditional on statewide year shocks and coarse regional temperature. Fixed effects are removed by alternating projections. Standard errors use a finite-sample CR1 correction, are clustered by HUC12, and confidence intervals use a t distribution with clusters minus one degrees of freedom.

### Drought definition and heterogeneous response

The primary drought measure is a 12-month standardized effective-precipitation index. For each HUC12, a zero-location gamma distribution is fit to 2000–2020 positive annual effective-precipitation values and transformed to standard-normal scores. **Drought is an index at or below −1.0**, approximately the NOAA/NWS moderate-drought threshold in the [Standardized Precipitation Index classification](https://www.cpc.ncep.noaa.gov/products/Drought/Monitoring/spi.shtml).

Because available precipitation is modeled effective precipitation rather than an independent meteorological gauge series, this is labeled an SPI proxy. The heterogeneity regression adds the drought indicator and `log(precipitation) × drought`; the interaction tests whether elasticity differs in drought years.

### Out-of-time validation

The predictive model trains on 2001–2013, with 2000 used only to construct the first lag, and is tested once on 2014–2020. It predicts log irrigation use from log effective precipitation, HUC4 temperature, lagged log irrigation use, and regularized HUC12 indicators. Ridge `alpha = 1.0` is fixed in advance. Log predictions are returned to MGD using a training-residual smearing correction, then compared with a last-observation baseline.

### Fixed robustness grid

Every pre-specified model is retained:

1. Drought-index cutoffs of −0.8 and −1.5.
2. Effective precipitation at or below each HUC12's 20th percentile.
3. Exclusion of units in the top 1% of unit-level mean irrigation use.
4. Alternative windows of 2000–2017 and 2005–2020.
5. The unbalanced sample requiring at least 15 positive paired years.

### Irrigation-response results

The source audit found **4,335 HUC12 columns** in the nine California HUC4 regions. The balanced-positive sample contains **1,169 HUC12 units and 24,549 unit-year observations**. Across **32,924 matched unit-years**, annualized monthly irrigation is reproduced as `Irrigation_wateruse − PPTeff` with maximum absolute discrepancy **0.000062 MGD**, confirming that the ambiguous annual file is total crop ET.

The estimated precipitation elasticity is **0.745** (HUC12-clustered SE **0.024**, 95% CI **0.698 to 0.792**, p = **1.86e-154**). Within this modeled panel, a 1% increase in effective precipitation is associated with a **0.745% increase** in irrigation consumptive use, conditional on HUC12 and year fixed effects and HUC4 temperature.

Under the SPI-proxy ≤ −1.0 definition, there are **3,929 drought observations** and **20,620 non-drought observations**. Raw mean irrigation is **16.318 MGD** in drought observations versus **13.140 MGD** otherwise, but the comparison reverses after scaling each observation by its unit mean: **0.918** in drought versus **1.016** otherwise. Estimated elasticity is **0.790** outside drought (95% CI **0.741 to 0.840**) and **0.841** during drought (95% CI **0.797 to 0.884**). The drought-minus-non-drought difference is **0.050** (95% CI **0.039 to 0.062**, p = **1.04e-16**).

For 2014–2020 validation, the ridge model has RMSE **5.046 MGD** and MAE **2.078 MGD**. The naive last-year baseline has RMSE **3.899 MGD** and MAE **1.758 MGD**; the model does not beat the baseline on both metrics.

Across the robustness grid, elasticity ranges from **0.625** (`window_2000_2017`) to **0.810** (`unbalanced_at_least_15_years`). Every requested specification remains in `results/irrigation_robustness.csv` regardless of significance.

![Irrigation drought response](../results/irrigation_drought_elasticity.png)

## Original correlation audit

The original **−0.52** was **not an irrigation correlation** and did not use `Irrigation_wateruse.csv`. Notebook cell 13 correlates San Joaquin HUC4 (`1804`) monthly public-supply use from `PS_HUC12_Tot_2000_2020.csv` with mean NOAA `PRCP (Inches)` from `Mercer.csv`, `Sanora.csv`, and `West_Point.csv`. Its exact Pearson correlation is **−0.517739693**. The notebook's rainfall loop inserts a leading zero month and omits December 2020.

Using the same legacy rainfall vector with the irrigation-consumptive-use series gives **−0.459764133**. Correctly aligning rainfall to calendar months gives **−0.672069975**. Inputs and outputs are saved in `results/correlation_audit.csv`.

## Previous 2018-policy analysis

The earlier analysis turns the 2000–2020 HUC12 monthly consumptive-use files and NOAA daily station files into an annual panel for nine California HUC4 watersheds. It describes how observed water use changed around California's 2018 water-conservation legislation while separating descriptive evidence from causal claims.

### Data construction

- `Irr_CU_HUC12_Tot_monthly_2000_2020.csv` contains irrigation consumptive use and `PS_HUC12_Tot_2000_2020.csv` contains public-supply use, both in MGD. The pipeline selects California HUC12 columns, treats `999` as missing, aggregates HUC12 values to HUC4 by month, and averages monthly totals by year.
- Three NOAA stations represent each HUC4. Annual station precipitation is adjusted for observed-day coverage; station-years below 70% coverage are excluded. Temperature uses TAVG when available and the TMAX/TMIN midpoint otherwise. Regional values are unweighted station means.
- `annual_panel.csv` is the watershed × year × category panel used by every model.

### Methods

The default before/after comparison is 2014–2017 versus 2018–2020. For every watershed, the pipeline reports pre/post means, absolute change, and percentage change. Uncertainty for the median percentage change uses a fixed-seed 20,000-resample percentile bootstrap over watersheds.

For rainfall adjustment, a separate pre-2018 OLS model, `total use = intercept + precipitation`, is fit for each watershed. The adjusted post-policy change is observed 2018–2020 use minus predicted use under actual rainfall. This is a weather-adjusted anomaly, not a causal effect.

California's [Department of Water Resources](https://water.ca.gov/Programs/Water-Use-And-Efficiency/2018-Water-Conservation-Legislation) describes SB 606 and AB 1668 as a statewide framework. Public supply is coded as more directly affected and irrigation as less directly affected, but irrigation is not untreated. The exploratory level regression is:

```text
annual MGD = public_supply × post_2018
           + precipitation + temperature
           + watershed-category fixed effects + year fixed effects
```

Standard errors are clustered by watershed. With only nine clusters, inference is fragile. The event study omits 2017, and the formal pre-trend diagnostic jointly tests 2010–2016 coefficients. Robustness includes a fake 2015 policy date, alternative start years, exclusion of bottom-decile precipitation years, alternative raw windows, and a pre-policy ridge prediction benchmark.

### Results

The default comparison covers nine watersheds. Median raw use changed by **2.70%** (watershed-bootstrap 95% CI **−0.69% to 20.02%**); two watersheds decreased and seven increased. The largest decrease was **Central Coastal (−7.80%)**, and the largest increase was **San Francisco Bay (20.69%)**. The median weather-adjusted post-policy anomaly was **8.71%** (95% bootstrap CI **−0.69% to 18.29%**).

The exploratory category DiD estimate is **−124.97 MGD** (watershed-clustered SE **63.28**, 95% CI **−248.99 to −0.94**, p = **0.0483**; nine clusters). The 2010–2016 pre-period joint test has p = **1.08e-07**, contradicting parallel trends. The 2015 placebo is **−51.78 MGD** (95% CI **−102.86 to −0.69**, p = **0.047**), so the falsification check fails. Dropping bottom precipitation-decile years (2007; 2013; 2020) gives **−99.78 MGD** (95% CI **−211.34 to 11.77**).

On the pre-policy holdout, the ridge model has RMSE **168.13 MGD** and MAE **114.29 MGD**. The last-year baseline has RMSE **168.35 MGD** and MAE **104.49 MGD**. The ridge model does not unambiguously beat the baseline.

Reproduce this earlier analysis with:

```bash
python -m water_policy.pipeline
python -m pytest
```

## Limitations

- The nine HUC4 watersheds are hydrologic aggregates, not counties, water agencies, or exact state-boundary clips.
- Three stations per HUC4 cannot capture all within-region climate variation. Coverage adjustment assumes missing days resemble observed days, and temperature is averaged from available stations.
- Water use is modeled rather than directly metered everywhere. Effective precipitation and irrigation consumptive use are jointly separated by the same soil-water-balance procedure.
- The positive irrigation elasticity does **not** establish that rainfall causes farmers to irrigate more; time-varying irrigated extent can move both quantities.
- The policy analysis has only three post years, including pandemic-affected 2020, and percentage changes can be sensitive to small baselines.
- A statewide law supplies no untreated California watershed. Public-supply-versus-irrigation comparisons do not make parallel trends plausible by construction, and event studies or placebos can reveal problems but cannot validate the assumption.

The irrigation-response estimates must be described as modeled physical associations. The policy before/after results are descriptive, rainfall residuals are model-adjusted associations, and the DiD is exploratory rather than causal.
