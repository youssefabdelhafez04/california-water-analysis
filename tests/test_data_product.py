import pandas as pd
import pytest

from water_policy.build_data_product import aggregate_dashboard_panel


def test_dashboard_aggregation_sums_huc12_and_preserves_weather():
    panel = pd.DataFrame(
        {
            "huc4": ["1804", "1804", "1804", "1804"],
            "watershed": ["San Joaquin"] * 4,
            "huc12": ["a", "b", "a", "b"],
            "year": [2000, 2000, 2001, 2001],
            "irrigation_mgd": [1.0, 2.0, 3.0, 4.0],
            "effective_precipitation_mgd": [4.0, 5.0, 6.0, 7.0],
        }
    )
    weather = pd.DataFrame(
        {
            "huc4": ["1804", "1804"],
            "year": [2000, 2001],
            "precipitation_in": [12.0, 14.0],
            "temperature_f": [60.0, 61.0],
            "precipitation_station_count": [3, 3],
        }
    )
    result = aggregate_dashboard_panel(panel, weather).set_index("year")
    assert result.loc[2000, "irrigation_mgd"] == pytest.approx(3.0)
    assert result.loc[2001, "effective_precipitation_mgd"] == pytest.approx(13.0)
    assert result.loc[2000, "huc12_count"] == 2
    assert result.loc[2001, "temperature_f"] == pytest.approx(61.0)
