# Public data download guide

The large raw inputs are intentionally excluded from Git. Keep the downloaded files in the project root with the exact names expected by the pipeline.

## USGS modeled water data

Download the HUC12 water-use files from the USGS data release [Crop irrigation consumptive water use and effective precipitation, 2000–2020](https://water.usgs.gov/catalog/datasets/649711fe-637f-4cc0-9020-09a21e691d88/).

Required local filenames:

```text
Irr_CU_HUC12_Tot_monthly_2000_2020.csv
PPTeff_HUC12_Tot_annual_2000_2020.csv
PS_HUC12_Tot_2000_2020.csv
Irrigation_wateruse.csv
PublicSupply_wateruse.csv
```

`Irr_CU_HUC12_Tot_monthly_2000_2020.csv` is the irrigation-consumptive-use outcome. The identity audit demonstrates that the ambiguously named annual `Irrigation_wateruse.csv` is total crop evapotranspiration and must not be substituted as the outcome.

## NOAA station observations

Download daily station CSVs from [NOAA Climate Data Online](https://www.ncei.noaa.gov/cdo-web/search) for the stations and date range used in the original notebooks. Select daily precipitation plus average/minimum/maximum temperature and retain NOAA's CSV export format. The pipeline's station-to-HUC4 mapping and expected filenames are defined in `water_policy/data.py`.

The repository's NOAA inputs use names such as:

```text
Barstow.csv
Big_Sur.csv
Cloverdale-Northern California Coastal.csv
...
West_Point.csv
```

## Watershed boundaries

Download California/HUC boundary geometry from the [USGS Watershed Boundary Dataset](https://www.usgs.gov/national-hydrography/watershed-boundary-dataset) or its linked National Map service. Export GeoJSON and save it as:

```text
b7974dff79ff41f0bd3cd4d8f248c6f4.json
```

## Rebuild

After all inputs are present:

```bash
python -m pip install -r requirements.txt
python -m water_policy.build_data_product
python -m pytest
```

This recreates `results/water_analysis.sqlite`, compact dashboard extracts, analysis CSVs, figures, validation checks, and the data-product manifest. Raw inputs and large generated extracts stay ignored; only compact results and the deployment database are versioned.
