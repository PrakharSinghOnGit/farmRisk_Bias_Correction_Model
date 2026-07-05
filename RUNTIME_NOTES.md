# Train-On-Demand Runtime Notes

This API intentionally wraps the current `village_forecast_combined.py` pipeline
without changing model behavior.

## Required Files

The API needs these files/folders beside `main.py`:

```text
village_forecast_combined.py
ERA_parquets/training_tmax_daily.parquet
ERA_parquets/training_tmin_daily.parquet
ERA_parquets/training_pcp_daily.parquet
IMD_parquets/imd_tmax_daily.parquet
IMD_parquets/imd_tmin_daily.parquet
IMD_parquets/imd_pcp_daily.parquet
OM_parquet/om_forecast_all.parquet
Elevation_Parquet/grid_elevation.parquet
```

These are not used by the current train-on-demand script and can be omitted from
runtime deployment without changing model predictions:

```text
Elevation_Parquet/train_tmax.parquet
Elevation_Parquet/train_tmin.parquet
Elevation_Parquet/train_pcp.parquet
Elevation_Parquet/test_tmax.parquet
Elevation_Parquet/test_tmin.parquet
Elevation_Parquet/test_pcp.parquet
OM_parquet/archive/
OM_parquet/forecast_archive_master.parquet
ERA_parquets/grid_ocean_points.csv
ERA_parquets/merge_qc_report.txt
README.md
training/comparison artifacts
```

## Safe Optimizations

Safe means predictions should not change.

- Remove files not listed in "Required Files".
- Do not deploy `.venv`; install from `requirements.txt`.
- Keep only the exact required parquet files.
- Recompress/rewrite parquets only if row values, dtypes, and columns remain
  identical.
- Partitioning required parquets by `lat/lon` can improve load time, but the API
  code should be validated after rewriting.

## Unsafe Optimizations

These can change the model and should be avoided if model behavior must remain
unchanged:

- Dropping old rows, even if they are from 1990.
- Removing any training dates.
- Downsampling rows.
- Rounding numeric values.
- Removing columns used by the script.
- Changing XGBoost parameters.
- Changing feature engineering.

The current model trains on all available rows for the 4 surrounding grid points
up to `TRAIN_END = 2025-12-31`. Old records influence the trained model, so
dropping them is a model change.

## Run

```bash
cd forcastModel
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Check health:

```bash
curl http://localhost:8000/health
```

Run inference:

```bash
curl -X POST http://localhost:8000/forecast \
  -H 'Content-Type: application/json' \
  -d '{"lat":22.3,"lon":72.6}'
```

Clear cache:

```bash
curl -X POST http://localhost:8000/cache/clear \
  -H 'Content-Type: application/json' \
  -d '{"key":"your-cache-clear-key"}'
```

## API Endpoints

```text
GET  /health
GET  /runtime-files
POST /forecast
POST /cache/clear
```

There is intentionally only one forecast endpoint: `POST /forecast`.

## Cache Behavior

The API caches by the 4 surrounding `0.25 degree` grid blocks, not by exact
latitude/longitude.

Example:

```text
lat/lon A -> surrounding blocks X -> generate forecast -> cache X
lat/lon B -> same surrounding blocks X -> return cached forecast
```

The cache is in memory and clears automatically when the server sees a new
calendar day. This matches the idea that daily forecasts should refresh at the
start of the next day.

Forecast responses include:

```text
cache_hit
cache_key
runtime_seconds
requested_lat
requested_lon
generated_for
```

If `cache_hit` is `true`, `generated_for` tells you which exact lat/lon first
created that cached block forecast.

Temporary CSV folders created by the model are deleted after the response is
read, so the server does not keep one result folder per lat/lon.

## Logs

By default, the verbose output from `village_forecast_combined.py` is suppressed
so Hugging Face logs do not fill with every training step.

Set this only while debugging:

```bash
LOG_MODEL_OUTPUT=true
```

Optional environment variables:

```text
OPEN_METEO_API_KEY
CACHE_CLEAR_KEY
MAX_CONCURRENT_FORECASTS
LOG_MODEL_OUTPUT
```
