import pandas as pd
import pytest

from water_policy.analysis import AnalysisWindow, assign_treatment, compare_windows


def test_configurable_windows_are_inclusive_and_compute_expected_change():
    panel = pd.DataFrame(
        {
            "watershed": ["A"] * 6 + ["B"] * 6,
            "year": [2015, 2016, 2017, 2018, 2019, 2020] * 2,
            "total_use_mgd": [8, 10, 12, 12, 14, 16, 20, 20, 20, 10, 10, 10],
        }
    )
    result = compare_windows(panel, AnalysisWindow(2015, 2017, 2018, 2020)).set_index("watershed")
    assert result.loc["A", "pre_mean_mgd"] == pytest.approx(10)
    assert result.loc["A", "post_mean_mgd"] == pytest.approx(14)
    assert result.loc["A", "absolute_change_mgd"] == pytest.approx(4)
    assert result.loc["A", "percent_change"] == pytest.approx(40)
    assert result.loc["B", "percent_change"] == pytest.approx(-50)


def test_incomplete_window_is_excluded_instead_of_silently_averaged():
    panel = pd.DataFrame(
        {
            "watershed": ["complete"] * 4 + ["missing"] * 3,
            "year": [2016, 2017, 2018, 2019, 2016, 2018, 2019],
            "total_use_mgd": [1, 1, 2, 2, 1, 2, 2],
        }
    )
    result = compare_windows(panel, AnalysisWindow(2016, 2017, 2018, 2019))
    assert result["watershed"].tolist() == ["complete"]


@pytest.mark.parametrize(
    ("category", "expected"),
    [("public_supply", 1), ("Public Supply", 1), ("public-supply", 1), ("irrigation", 0)],
)
def test_treatment_assignment(category, expected):
    assert assign_treatment(category) == expected


def test_treatment_assignment_rejects_unsupported_categories():
    with pytest.raises(ValueError, match="unknown water-use category"):
        assign_treatment("industrial")


@pytest.mark.parametrize(
    "window",
    [
        (2018, 2017, 2018, 2020),
        (2014, 2018, 2018, 2020),
        (2014, 2017, 2021, 2020),
    ],
)
def test_invalid_windows_are_rejected(window):
    with pytest.raises(ValueError):
        AnalysisWindow(*window)
