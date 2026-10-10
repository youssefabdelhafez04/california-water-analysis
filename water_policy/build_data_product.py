"""Build the complete reproducible data product from raw local source files.

Run from the project directory with::

    python -m water_policy.build_data_product
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from pathlib import Path

import pandas as pd

from .analysis import AnalysisWindow
from .correlation_audit import correlation_audit
from .data import WATERSHEDS, build_annual_panel, load_weather
from .irrigation_response import build_huc12_panel, run as run_irrigation
from .pipeline import run as run_policy


DATABASE_NAME = "water_analysis.sqlite"
RAW_REQUIRED = [
    "Irr_CU_HUC12_Tot_monthly_2000_2020.csv",
    "PS_HUC12_Tot_2000_2020.csv",
    "PPTeff_HUC12_Tot_annual_2000_2020.csv",
    "Irrigation_wateruse.csv",
    "b7974dff79ff41f0bd3cd4d8f248c6f4.json",
]


def validate_inputs(data_dir: Path) -> None:
    missing = [name for name in RAW_REQUIRED if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing required raw files: {missing}")


def aggregate_dashboard_panel(full_huc12: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Aggregate cleaned HUC12 quantities to a small HUC4-year dashboard table."""

    grouped = full_huc12.groupby(["huc4", "watershed", "year"])
    irrigation = grouped["irrigation_mgd"].sum(min_count=1).rename("irrigation_mgd")
    precipitation = grouped["effective_precipitation_mgd"].sum(min_count=1).rename(
        "effective_precipitation_mgd"
    )
    counts = grouped["irrigation_mgd"].count().rename("huc12_count")
    dashboard = pd.concat([irrigation, precipitation, counts], axis=1).reset_index()
    weather_columns = weather[
        ["huc4", "year", "precipitation_in", "temperature_f", "precipitation_station_count"]
    ]
    dashboard = dashboard.merge(weather_columns, on=["huc4", "year"], validate="one_to_one")
    return dashboard.sort_values(["year", "huc4"]).reset_index(drop=True)


def _perpendicular_distance(point, start, end) -> float:
    if start == end:
        return math.dist(point, start)
    x, y = point
    x1, y1 = start
    x2, y2 = end
    numerator = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
    return numerator / math.hypot(y2 - y1, x2 - x1)


def _rdp(points: list[list[float]], tolerance: float) -> list[list[float]]:
    if len(points) <= 2:
        return points
    distances = [_perpendicular_distance(point, points[0], points[-1]) for point in points[1:-1]]
    if not distances or max(distances) <= tolerance:
        return [points[0], points[-1]]
    index = distances.index(max(distances)) + 1
    return _rdp(points[: index + 1], tolerance)[:-1] + _rdp(points[index:], tolerance)


def _web_mercator_to_lon_lat(point: list[float]) -> list[float]:
    radius = 6_378_137.0
    lon = math.degrees(point[0] / radius)
    lat = math.degrees(2 * math.atan(math.exp(point[1] / radius)) - math.pi / 2)
    return [round(lon, 6), round(lat, 6)]


def build_dashboard_geojson(source_path: Path, output_path: Path) -> dict:
    """Extract and simplify the nine HUC4 polygons for browser delivery."""

    source = json.loads(source_path.read_text())
    features = []
    for item in source["layers"][0]["features"]:
        huc4 = str(item["attributes"].get("huc4"))
        if huc4 not in WATERSHEDS:
            continue
        rings = []
        for ring in item["geometry"]["rings"]:
            closed = ring if ring[0] == ring[-1] else [*ring, ring[0]]
            simplified = _rdp(closed, tolerance=2_500.0)
            if simplified[0] != simplified[-1]:
                simplified.append(simplified[0])
            if len(simplified) >= 4:
                rings.append([_web_mercator_to_lon_lat(point) for point in simplified])
        features.append(
            {
                "type": "Feature",
                "id": huc4,
                "properties": {"huc4": huc4, "watershed": WATERSHEDS[huc4]},
                "geometry": {"type": "Polygon", "coordinates": rings},
            }
        )
    geojson = {"type": "FeatureCollection", "features": features}
    output_path.write_text(json.dumps(geojson, separators=(",", ":")))
    return geojson


def write_database(
    database_path: Path,
    huc12_panel: pd.DataFrame,
    huc4_category_panel: pd.DataFrame,
    dashboard: pd.DataFrame,
    correlations: pd.DataFrame,
    results_dir: Path,
) -> dict[str, int]:
    if database_path.exists():
        database_path.unlink()
    tables = {
        "irrigation_huc12_annual": huc12_panel.drop(columns=["positive_pair"], errors="ignore"),
        "water_huc4_annual_category": huc4_category_panel,
        "dashboard_huc4_year": dashboard,
        "correlation_audit": correlations,
        "irrigation_elasticity_results": pd.read_csv(results_dir / "irrigation_elasticity.csv"),
        "irrigation_drought_results": pd.read_csv(results_dir / "irrigation_drought_response.csv"),
        "irrigation_robustness_results": pd.read_csv(results_dir / "irrigation_robustness.csv"),
        "irrigation_validation_metrics": pd.read_csv(results_dir / "irrigation_validation_metrics.csv"),
        "policy_before_after_results": pd.read_csv(results_dir / "watershed_before_after.csv"),
        "policy_did_results": pd.read_csv(results_dir / "did_results.csv"),
    }
    with sqlite3.connect(database_path) as connection:
        for name, frame in tables.items():
            frame.to_sql(name, connection, if_exists="replace", index=False)
        connection.execute(
            "CREATE INDEX idx_irrigation_huc12_year ON irrigation_huc12_annual(huc12, year)"
        )
        connection.execute(
            "CREATE INDEX idx_huc4_category_year ON water_huc4_annual_category(huc4, category, year)"
        )
        connection.execute("CREATE INDEX idx_dashboard_year ON dashboard_huc4_year(year, huc4)")
    return {name: len(frame) for name, frame in tables.items()}


def update_readme_correlation(readme: Path, correlations: pd.DataFrame) -> None:
    text = readme.read_text()
    start = "<!-- CORRELATION_AUDIT_START -->"
    end = "<!-- CORRELATION_AUDIT_END -->"
    values = correlations.set_index("analysis")["pearson_correlation"]
    generated = f"""{start}
## Original correlation audit

The original **−0.52** was **not an irrigation correlation** and did not use `Irrigation_wateruse.csv`. Notebook cell 13 correlates San Joaquin HUC4 (`1804`) monthly public-supply use from `PS_HUC12_Tot_2000_2020.csv` with mean NOAA `PRCP (Inches)` from `Mercer.csv`, `Sanora.csv`, and `West_Point.csv`. Its exact Pearson correlation is **{values['original_reported_minus_0_52']:.9f}**. The notebook's rainfall loop inserts a leading zero month and omits December 2020.

Using the same legacy rainfall vector with the actual irrigation-consumptive-use series from `Irr_CU_HUC12_Tot_monthly_2000_2020.csv` gives **{values['like_for_like_correct_irrigation']:.9f}**. Correctly aligning rainfall to calendar months gives **{values['calendar_aligned_correct_irrigation']:.9f}**. All three auditable inputs and outputs are saved in `results/correlation_audit.csv`.
{end}"""
    before, remainder = text.split(start, 1)
    _, after = remainder.split(end, 1)
    readme.write_text(before + generated + after)


def run(data_dir: Path, results_dir: Path, readme: Path) -> None:
    validate_inputs(data_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Both analysis modules regenerate their CSVs, figures, and README sections.
    run_irrigation(data_dir, results_dir, readme)
    run_policy(data_dir, results_dir, AnalysisWindow(), readme)

    full_huc12, availability = build_huc12_panel(data_dir)
    huc4_category = build_annual_panel(data_dir)
    weather = load_weather(data_dir)
    dashboard = aggregate_dashboard_panel(full_huc12, weather)
    correlations = correlation_audit(data_dir)

    dashboard.to_csv(results_dir / "dashboard_huc4_year.csv", index=False)
    correlations.to_csv(results_dir / "correlation_audit.csv", index=False)
    availability.to_csv(results_dir / "irrigation_data_availability.csv", index=False)
    geometry = build_dashboard_geojson(
        data_dir / "b7974dff79ff41f0bd3cd4d8f248c6f4.json",
        results_dir / "dashboard_huc4.geojson",
    )
    row_counts = write_database(
        results_dir / DATABASE_NAME,
        full_huc12,
        huc4_category,
        dashboard,
        correlations,
        results_dir,
    )
    manifest = {
        "command": "python -m water_policy.build_data_product",
        "database": str(results_dir / DATABASE_NAME),
        "database_table_rows": row_counts,
        "dashboard_rows": len(dashboard),
        "dashboard_geojson_features": len(geometry["features"]),
        "correlation_rows": len(correlations),
    }
    (results_dir / "data_product_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    update_readme_correlation(readme, correlations)


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
