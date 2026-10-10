"""Reproduce and audit the notebook's reported -0.52 correlation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import STATIONS, load_monthly_water


def monthly_station_rain(data_dir: Path, huc4: str, *, legacy_shift: bool) -> np.ndarray:
    station_series = []
    dates = None
    for filename in STATIONS[huc4]:
        daily = pd.read_csv(data_dir / filename, skiprows=1)
        daily["Date"] = pd.to_datetime(daily["Date"], errors="coerce")
        daily = daily[daily["Date"].between("2000-01-01", "2020-12-31")].copy()
        rain = pd.to_numeric(daily["PRCP (Inches)"], errors="coerce").fillna(0.0)
        station_series.append(rain.to_numpy())
        dates = daily["Date"].reset_index(drop=True)
    regional = np.mean(np.vstack(station_series), axis=0)
    if legacy_shift:
        # Exact notebook behavior: append at each first-of-month before adding
        # that day's rainfall. This creates a leading zero and omits Dec. 2020.
        result: list[float] = []
        total = 0.0
        for date, value in zip(dates, regional, strict=True):
            if date.day == 1:
                result.append(total)
                total = float(value)
            else:
                total += float(value)
        return np.asarray(result)
    frame = pd.DataFrame({"date": dates, "rain": regional})
    return frame.groupby(frame["date"].dt.to_period("M"))["rain"].sum().to_numpy()


def correlation_audit(data_dir: Path) -> pd.DataFrame:
    huc4 = "1804"
    public = load_monthly_water(data_dir / "PS_HUC12_Tot_2000_2020.csv", "public_supply")
    irrigation = load_monthly_water(
        data_dir / "Irr_CU_HUC12_Tot_monthly_2000_2020.csv", "irrigation"
    )
    public_values = public.loc[public["huc4"].eq(huc4), "use_mgd"].to_numpy()
    irrigation_values = irrigation.loc[irrigation["huc4"].eq(huc4), "use_mgd"].to_numpy()
    legacy_rain = monthly_station_rain(data_dir, huc4, legacy_shift=True)
    aligned_rain = monthly_station_rain(data_dir, huc4, legacy_shift=False)
    if not (len(public_values) == len(irrigation_values) == len(legacy_rain) == len(aligned_rain) == 252):
        raise ValueError("correlation inputs must each contain 252 months")
    rows = [
        {
            "analysis": "original_reported_minus_0_52",
            "water_file": "PS_HUC12_Tot_2000_2020.csv",
            "water_series": "public supply total MGD summed from HUC12 to HUC4 1804",
            "rain_files": ";".join(STATIONS[huc4]),
            "rain_series": "mean daily PRCP (Inches) across 3 stations, legacy month shift",
            "months": 252,
            "pearson_correlation": np.corrcoef(public_values, legacy_rain)[0, 1],
        },
        {
            "analysis": "like_for_like_correct_irrigation",
            "water_file": "Irr_CU_HUC12_Tot_monthly_2000_2020.csv",
            "water_series": "irrigation consumptive use MGD summed from HUC12 to HUC4 1804",
            "rain_files": ";".join(STATIONS[huc4]),
            "rain_series": "mean daily PRCP (Inches) across 3 stations, legacy month shift",
            "months": 252,
            "pearson_correlation": np.corrcoef(irrigation_values, legacy_rain)[0, 1],
        },
        {
            "analysis": "calendar_aligned_correct_irrigation",
            "water_file": "Irr_CU_HUC12_Tot_monthly_2000_2020.csv",
            "water_series": "irrigation consumptive use MGD summed from HUC12 to HUC4 1804",
            "rain_files": ";".join(STATIONS[huc4]),
            "rain_series": "mean daily PRCP (Inches) across 3 stations, calendar-month sum",
            "months": 252,
            "pearson_correlation": np.corrcoef(irrigation_values, aligned_rain)[0, 1],
        },
    ]
    return pd.DataFrame(rows)
