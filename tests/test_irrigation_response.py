import numpy as np
import pandas as pd
import pytest

from water_policy.irrigation_response import (
    annualize_monthly_huc12,
    clean_water_values,
    fixed_effect_regression,
    select_balanced_positive,
    standardized_precipitation_proxy,
)


def test_clean_water_values_coerces_numbers_and_removes_999_sentinel():
    raw = pd.DataFrame({"a": ["1.5", "999", "bad"], "b": [2, 3, 4]})
    cleaned = clean_water_values(raw)
    assert cleaned.loc[0, "a"] == pytest.approx(1.5)
    assert np.isnan(cleaned.loc[1, "a"])
    assert np.isnan(cleaned.loc[2, "a"])
    assert cleaned["b"].tolist() == [2, 3, 4]


def test_annualize_monthly_huc12_uses_calendar_day_weights():
    monthly = pd.DataFrame(
        {
            "Year": [2020] * 12,
            "Month": list(range(1, 13)),
            "180400000001": [10.0, 20.0] + [0.0] * 10,
            "180400000002": [999, 5.0] + [5.0] * 10,
        }
    )
    annual = annualize_monthly_huc12(monthly)
    expected = (10 * 31 + 20 * 29) / 366
    assert annual.loc[0, "180400000001"] == pytest.approx(expected)
    assert np.isnan(annual.loc[0, "180400000002"])


def test_spi_proxy_preserves_precipitation_order_and_is_centered():
    values = pd.Series([1, 2, 3, 4, 6, 9, 13, 18], dtype=float)
    spi = standardized_precipitation_proxy(values)
    assert spi.is_monotonic_increasing
    assert abs(spi.mean()) < 0.15


def test_balanced_positive_selection_requires_every_requested_year():
    panel = pd.DataFrame(
        {
            "huc12": ["good"] * 3 + ["zero"] * 3 + ["missing"] * 2,
            "huc4": ["1801"] * 8,
            "watershed": ["Northern Coastal"] * 8,
            "year": [2000, 2001, 2002, 2000, 2001, 2002, 2000, 2002],
            "irrigation_mgd": [1, 2, 3, 1, 0, 3, 1, 3],
            "effective_precipitation_mgd": [3, 2, 1, 3, 2, 1, 3, 1],
            "temperature_f": [60] * 8,
            "temperature_station_count": [3] * 8,
            "positive_pair": [True, True, True, True, False, True, True, True],
        }
    )
    selected = select_balanced_positive(panel, 2000, 2002)
    assert selected["huc12"].unique().tolist() == ["good"]


def test_fixed_effect_estimator_recovers_known_within_elasticity():
    rows = []
    for unit_index, unit in enumerate(["a", "b", "c", "d"]):
        for year in range(2000, 2006):
            log_precip = 0.2 * year + 0.3 * unit_index + 0.07 * unit_index * (year - 2000)
            rows.append(
                {
                    "huc12": unit,
                    "year": year,
                    "log_precipitation": log_precip,
                    "temperature_f": 50 + unit_index + (year - 2000) + 0.02 * unit_index * (year - 2000) ** 2,
                    "log_irrigation": -0.4 * log_precip + 2 * unit_index + 0.5 * (year - 2000),
                }
            )
    result, _ = fixed_effect_regression(
        pd.DataFrame(rows),
        "log_irrigation",
        ["log_precipitation", "temperature_f"],
        specification="synthetic",
    )
    estimate = result.set_index("term").loc["log_precipitation", "estimate"]
    assert estimate == pytest.approx(-0.4, abs=1e-8)
