"""Prompt components for the California water data agent."""

SCHEMA_GUIDE = """
Database guide

Core data tables

1. dashboard_huc4_year: one row per HUC4 watershed and year, 2000-2020.
   - huc4: four-character hydrologic region code (TEXT; preserve leading zeroes).
   - watershed: HUC4 watershed name (TEXT).
   - year: calendar year (INTEGER, 2000-2020).
   - irrigation_mgd: total modeled irrigation consumptive use across included HUC12s,
     in million gallons per day (MGD).
   - effective_precipitation_mgd: effective precipitation over irrigated land expressed
     as an MGD-equivalent flow.
   - huc12_count: number of contributing HUC12 subwatersheds.
   - precipitation_in: annual NOAA station precipitation, inches.
   - temperature_f: annual mean NOAA station temperature, degrees Fahrenheit.
   - precipitation_station_count: number of NOAA precipitation stations used.

2. water_huc4_annual_category: one row per watershed, year, and water-use category,
   2000-2020.
   - watershed, huc4, year: same meanings as dashboard_huc4_year.
   - category: exactly 'irrigation' or 'public_supply' (underscore, lowercase).
   - annual_avg_use_mgd: annual average water use, MGD, for that category.
   - precipitation_in: annual precipitation, inches.
   - temperature_f: annual mean temperature, degrees Fahrenheit.
   - precipitation_station_count, temperature_station_count: contributing station counts.

3. irrigation_huc12_annual: one row per HUC12 subwatershed and year, 2000-2020.
   - year: calendar year.
   - huc12: 12-character hydrologic unit code (TEXT).
   - irrigation_mgd: modeled irrigation consumptive use, MGD; may be NULL.
   - effective_precipitation_mgd: effective precipitation, MGD equivalent; may be NULL.
   - huc4, watershed: containing HUC4 code and name.
   - temperature_f: containing HUC4 annual mean temperature, degrees Fahrenheit.
   - temperature_station_count: contributing temperature station count.

Audit and model-result tables

4. correlation_audit: one row per correlation calculation.
   - analysis: calculation identifier. Valid identifiers include
     'original_reported_minus_0_52', 'like_for_like_correct_irrigation', and
     'calendar_aligned_correct_irrigation'.
   - water_file, water_series, rain_files, rain_series: source/provenance descriptions.
   - months: number of monthly observations.
   - pearson_correlation: Pearson correlation coefficient.

5. irrigation_elasticity_results: regression output by specification and term.
   - specification, term: model and coefficient identifiers.
   - estimate, clustered_se, ci_95_low, ci_95_high, p_value: coefficient statistics.
   - observations, huc12_clusters: sample sizes.
   - start_year, end_year: inclusive model window.

6. irrigation_drought_results: drought-interaction regression output.
   - specification, drought_definition: model identifiers.
   - non_drought_elasticity, non_drought_clustered_se, non_drought_ci_95_low,
     non_drought_ci_95_high: non-drought estimate and uncertainty.
   - drought_elasticity, drought_clustered_se, drought_ci_95_low,
     drought_ci_95_high, drought_elasticity_p_value: drought estimate and uncertainty.
   - elasticity_difference, difference_clustered_se, difference_ci_95_low,
     difference_ci_95_high, difference_p_value: drought minus non-drought contrast.
   - drought_observations, non_drought_observations, observations, huc12_clusters:
     sample sizes.
   - start_year, end_year: inclusive model window.

7. irrigation_robustness_results: robustness specifications. It contains every column
   in irrigation_drought_results plus overall_elasticity, overall_clustered_se,
   overall_ci_95_low, overall_ci_95_high, overall_p_value, and
   outlier_cutoff_mean_irrigation_mgd.

8. irrigation_validation_metrics: one row per prediction model.
   - model, train_window, test_window: model and period identifiers.
   - rmse_mgd, mae_mgd: level-error metrics in MGD.
   - log_rmse: log-scale RMSE.
   - test_observations, huc12_units: test sample sizes.
   - smearing_factor: retransformation factor.

9. policy_before_after_results: one row per watershed.
   - watershed: HUC4 watershed name.
   - pre_mean_mgd, post_mean_mgd, absolute_change_mgd, percent_change: raw comparison.
   - rain_adjusted_change_mgd, rain_adjusted_percent_change: rainfall-adjusted comparison.
   - pre_2018_rain_slope_mgd_per_in, pre_2018_r_squared: pre-2018 rainfall model terms.

10. policy_did_results: one row per difference-in-differences specification.
    - specification, policy_year: model identifiers.
    - effect_mgd, clustered_se_mgd, ci_95_low_mgd, ci_95_high_mgd, p_value:
      effect estimate and uncertainty.
    - watershed_clusters, observations: sample sizes.

Join rules

- Join the HUC4 tables on huc4 and year. Watershed may be included as a consistency key.
- Join irrigation_huc12_annual to an HUC4 table on huc4 and year, then aggregate HUC12
  rows before comparing with an HUC4 total.
- Never join only on watershed when year is part of the question.
- Filtering category data must use the stored values 'irrigation' and 'public_supply'.
- The result tables generally do not need joins unless a question explicitly combines them.

Query rules

- Return exactly the columns and rows requested. Do not add identifiers, component values,
  counts, or diagnostic columns unless the question asks for them.
- Use dashboard_huc4_year for HUC4 irrigation, effective precipitation, temperature,
  precipitation, and HUC12-count questions unless a category comparison is requested.
- Use water_huc4_annual_category when the question names irrigation/public-supply categories.
- Keep year filters inside a window-function input when the prior year is needed.
""".strip()


FEW_SHOT_EXAMPLES = """
Examples

Question: What was Northern Coastal precipitation in 2008?
SQL: SELECT precipitation_in FROM dashboard_huc4_year
     WHERE watershed = 'Northern Coastal' AND year = 2008

Question: What was average Southern Mojave irrigation MGD from 2004 through 2012?
SQL: SELECT AVG(irrigation_mgd) FROM dashboard_huc4_year
     WHERE watershed = 'Southern Mojave' AND year BETWEEN 2004 AND 2012

Question: How much did Central Coastal public-supply use change from 2012 to 2016?
SQL: SELECT MAX(CASE WHEN year = 2016 THEN annual_avg_use_mgd END)
          - MAX(CASE WHEN year = 2012 THEN annual_avg_use_mgd END) AS change_mgd
     FROM water_huc4_annual_category
     WHERE watershed = 'Central Coastal' AND category = 'public_supply'
       AND year IN (2012, 2016)

Question: For Tulare-Buena Vista in 2011, return temperature and public-supply use.
SQL: SELECT d.temperature_f, c.annual_avg_use_mgd
     FROM dashboard_huc4_year AS d
     JOIN water_huc4_annual_category AS c
       ON c.huc4 = d.huc4 AND c.year = d.year AND c.category = 'public_supply'
     WHERE d.watershed = 'Tulare-Buena Vista' AND d.year = 2011
""".strip()
