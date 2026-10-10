"""Build the annual HUC4 panel from the original wide water and NOAA files."""

from __future__ import annotations

import csv
from calendar import isleap
from pathlib import Path

import numpy as np
import pandas as pd


WATERSHEDS = {
    "1801": "Northern Coastal",
    "1802": "Sacramento",
    "1803": "Tulare-Buena Vista",
    "1804": "San Joaquin",
    "1805": "San Francisco Bay",
    "1806": "Central Coastal",
    "1807": "Southern Coastal",
    "1809": "Northern Mojave",
    "1810": "Southern Mojave",
}

STATIONS = {
    "1801": [
        "Cloverdale-Northern California Coastal.csv",
        "Eureka-Northern California Coastal.csv",
        "Fort Bragg-Northern California Coastal.csv",
    ],
    "1802": ["Jess Valley - Sacramento.csv", "McCloud - Sacramento.csv", "Quincy, Sacramento.csv"],
    "1803": ["Coalinga.csv", "Fresno.csv", "Visalia.csv"],
    "1804": ["Mercer.csv", "Sanora.csv", "West_Point.csv"],
    "1805": [
        "Fremont - San Fransisco.csv",
        "Mount Hamilton - San Francisco Bay.csv",
        "Richmond - San Fransisco.csv",
    ],
    "1806": ["Big_Sur.csv", "Santa_Maria.csv", "Watsonville.csv"],
    "1807": ["Long Beach.csv", "Oceanside.csv", "San_Diego.csv"],
    "1809": ["Barstow.csv", "Death_Valley.csv", "Victorville.csv"],
    "1810": ["Iron_Mountain.csv", "Niland.csv", "Randsburg.csv"],
}


def _selected_columns(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        header = next(csv.reader(handle))
    return [c for c in header if c in {"Year", "Month"} or c[:4] in WATERSHEDS]


def load_monthly_water(path: Path, category: str) -> pd.DataFrame:
    """Aggregate the original HUC12 monthly MGD columns to HUC4."""

    usecols = _selected_columns(path)
    raw = pd.read_csv(path, usecols=usecols)
    value_cols = [c for c in usecols if c not in {"Year", "Month"}]
    values = raw[value_cols].apply(pd.to_numeric, errors="coerce")
    # The source uses 999 as a missing-data sentinel in areas without estimates.
    values = values.mask(values.eq(999))
    frames = []
    for huc4, name in WATERSHEDS.items():
        columns = [c for c in value_cols if c.startswith(huc4)]
        frames.append(
            pd.DataFrame(
                {
                    "watershed": name,
                    "huc4": huc4,
                    "year": raw["Year"].astype(int),
                    "month": raw["Month"].astype(int),
                    "category": category,
                    "use_mgd": values[columns].sum(axis=1, min_count=1),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def load_weather(data_dir: Path, min_coverage: float = 0.70) -> pd.DataFrame:
    """Create annual regional weather from three NOAA stations per HUC4.

    Annual station precipitation is coverage-adjusted by expected/observed
    days, then stations are averaged. Temperature uses TAVG when present and
    the midpoint of TMAX/TMIN otherwise. Station-years below ``min_coverage``
    are excluded for the affected measure.
    """

    records: list[pd.DataFrame] = []
    for huc4, files in STATIONS.items():
        for filename in files:
            daily = pd.read_csv(data_dir / filename, skiprows=1)
            daily["Date"] = pd.to_datetime(daily["Date"], errors="coerce")
            daily = daily[daily["Date"].dt.year.between(2000, 2020)].copy()
            daily["year"] = daily["Date"].dt.year.astype(int)
            tavg = pd.to_numeric(daily.get("TAVG (Degrees Fahrenheit)"), errors="coerce")
            midpoint = (
                pd.to_numeric(daily.get("TMAX (Degrees Fahrenheit)"), errors="coerce")
                + pd.to_numeric(daily.get("TMIN (Degrees Fahrenheit)"), errors="coerce")
            ) / 2
            daily["temperature_f"] = tavg.fillna(midpoint)
            daily["precipitation_in"] = pd.to_numeric(daily.get("PRCP (Inches)"), errors="coerce")

            annual = daily.groupby("year").agg(
                precipitation_sum=("precipitation_in", "sum"),
                precipitation_days=("precipitation_in", "count"),
                temperature_f=("temperature_f", "mean"),
                temperature_days=("temperature_f", "count"),
            )
            annual["expected_days"] = [366 if isleap(int(y)) else 365 for y in annual.index]
            precip_ok = annual["precipitation_days"] / annual["expected_days"] >= min_coverage
            temp_ok = annual["temperature_days"] / annual["expected_days"] >= min_coverage
            annual["precipitation_in"] = np.where(
                precip_ok,
                annual["precipitation_sum"] * annual["expected_days"] / annual["precipitation_days"],
                np.nan,
            )
            annual["temperature_f"] = annual["temperature_f"].where(temp_ok)
            annual["huc4"] = huc4
            annual["station"] = filename.removesuffix(".csv")
            records.append(annual.reset_index())

    station_years = pd.concat(records, ignore_index=True)
    regional = (
        station_years.groupby(["huc4", "year"], as_index=False)
        .agg(
            precipitation_in=("precipitation_in", "mean"),
            temperature_f=("temperature_f", "mean"),
            precipitation_station_count=("precipitation_in", "count"),
            temperature_station_count=("temperature_f", "count"),
        )
    )
    regional["watershed"] = regional["huc4"].map(WATERSHEDS)
    return regional


def build_annual_panel(data_dir: Path) -> pd.DataFrame:
    irrigation = load_monthly_water(data_dir / "Irr_CU_HUC12_Tot_monthly_2000_2020.csv", "irrigation")
    public = load_monthly_water(data_dir / "PS_HUC12_Tot_2000_2020.csv", "public_supply")
    monthly = pd.concat([irrigation, public], ignore_index=True)
    annual = (
        monthly.groupby(["watershed", "huc4", "year", "category"], as_index=False)["use_mgd"]
        .mean()
        .rename(columns={"use_mgd": "annual_avg_use_mgd"})
    )
    weather = load_weather(data_dir)
    panel = annual.merge(weather, on=["watershed", "huc4", "year"], how="left", validate="many_to_one")
    if panel[["annual_avg_use_mgd", "precipitation_in", "temperature_f"]].isna().any().any():
        missing = panel[panel[["annual_avg_use_mgd", "precipitation_in", "temperature_f"]].isna().any(axis=1)]
        raise ValueError(f"annual panel has missing core values:\n{missing.to_string(index=False)}")
    return panel.sort_values(["watershed", "category", "year"]).reset_index(drop=True)
