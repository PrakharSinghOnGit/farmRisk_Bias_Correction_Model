"""
Village Forecast Service — Cached Per-Grid Bias Correction
============================================================
Callable service version of the combined train+forecast pipeline.

Instead of a CLI script that writes a CSV, this module exposes:

    get_village_forecast(lat, lon) -> dict   (JSON-serializable)

Key changes vs. the original combined script:

  1. FUNCTION, NOT CSV
     `get_village_forecast(lat, lon)` returns a plain dict (safe to
     `json.dumps`). Nothing is written to disk as a "result" file.

  2. MODEL CACHE ACROSS REQUESTS
     Per-grid models (the expensive part — parquet scan + XGBoost fit)
     are cached to disk under MODEL_CACHE_DIR, keyed by (variable, grid
     lat, grid lon). Since the underlying grid is fixed at 0.25°, any
     two villages that share a grid corner reuse that grid's model
     instead of retraining it. An in-process memory cache also avoids
     re-reading from disk within the same run. Cache entries expire
     after CACHE_TTL_SECONDS so models eventually refresh.

  3. CONSTANTS UP TOP
     All directory paths, URLs, and the API key are declared as module
     constants at the top of the file — nothing is buried in argparse.

  4. STRUCTURED ERROR / SUCCESS RESPONSES
     Every call returns {"success": True, ...} or
     {"success": False, "error": {"type": ..., "message": ...}}.
     Partial failures (e.g. one variable couldn't train) are reported
     per-variable via "model_status" rather than crashing the request.

Usage
-----
    from village_forecast_service import get_village_forecast
    result = get_village_forecast(22.3, 72.6)
    print(json.dumps(result, indent=2))
"""

import os
import time
import json
import warnings
from datetime import date, datetime

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

try:
    import joblib
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False
    import pickle


# ==============================================================================
# CONSTANTS — everything path/key/URL related lives here
# ==============================================================================

# --- Data directories (inputs) ---
# Resolved relative to THIS FILE's location, not the process's working
# directory. Bare "./training_data" style paths silently break the moment
# this module is imported/called from a different CWD (a service, a
# notebook, another script's folder) -- every read then finds nothing, and
# every grid looks like "insufficient data" even though nothing is actually
# wrong with the data or the model params. If your data lives somewhere else
# entirely, hardcode absolute paths here instead.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

ERA5_DIR = os.path.join(DATA_DIR, "training_data")
FORECAST_PARQUET = os.path.join(DATA_DIR, "om_forecast_all.parquet")
IMD_DIR = os.path.join(DATA_DIR, "IMD_parquets")
ELEV_PARQUET = os.path.join(DATA_DIR, "grid_elevation.parquet")

# --- Model cache (replaces ./models_pergrid disk-writes from the old script) ---
MODEL_CACHE_DIR = os.path.join(MODELS_DIR, "models_cache_pergrid")
CACHE_TTL_SECONDS = 7 * 24 * 3600  # retrain a grid's model once a week

# --- Open-Meteo API ---
# NOTE: prefer sourcing this from an environment variable in production,
# e.g. OPEN_METEO_API_KEY = os.environ.get("OPEN_METEO_API_KEY", "")
OPEN_METEO_API_KEY = "XLv82nM2BPBe6qVe"
LIVE_FORECAST_URL = "https://customer-api.open-meteo.com/v1/forecast"
FREE_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ELEVATION_URL = "https://api.opentopodata.org/v1/srtm90m"
DAILY_PARAMS = "temperature_2m_max,temperature_2m_min,precipitation_sum"
FORECAST_DAYS = 16
PAST_BUFFER_DAYS = 8  # lead-in days needed for the 7-day rolling feature
API_RETRY_ATTEMPTS = 5

# --- Modeling constants ---
RAIN_THRESHOLD = 1.0
TRAIN_END = "2025-12-31"
GRID_STEP = 0.25
GRID_OFFSET = 0.125
MIN_TRAIN_ROWS = 100
MIN_RAIN_ROWS = 20

XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "random_state": 42,
    "verbosity": 0,
}

TEMP_FEATURES = [
    "om", "elevation",
    "doy_sin", "doy_cos", "month",
    "season_monsoon", "season_premonsoon", "season_postmonsoon", "season_winter",
    "om_roll7", "om_clim_anom",
    "source",
]
RAIN_CLS_FEATURES = TEMP_FEATURES + ["rain_om"]
RAIN_REG_FEATURES = ["om_log"] + TEMP_FEATURES

VAR_KEYS = ("tmax", "tmin", "pcp")


# ==============================================================================
# ERRORS — typed exceptions so the top-level function can build a clean response
# ==============================================================================

class VillageForecastError(Exception):
    """Base class for all handled errors in this module."""
    error_type = "internal_error"


class InvalidLocationError(VillageForecastError):
    error_type = "invalid_location"


class ForecastFetchError(VillageForecastError):
    error_type = "forecast_fetch_failed"


class ElevationFetchError(VillageForecastError):
    error_type = "elevation_fetch_failed"


# ==============================================================================
# SMALL UTILITIES
# ==============================================================================

def _log(verbose, msg):
    if verbose:
        print(msg)


def _validate_location(lat, lon):
    try:
        lat = float(lat)
        lon = float(lon)
    except (TypeError, ValueError):
        raise InvalidLocationError(f"lat/lon must be numeric, got lat={lat!r} lon={lon!r}")
    if not (-90.0 <= lat <= 90.0):
        raise InvalidLocationError(f"lat {lat} out of range [-90, 90]")
    if not (-180.0 <= lon <= 180.0):
        raise InvalidLocationError(f"lon {lon} out of range [-180, 180]")
    return lat, lon


# ==============================================================================
# GRID GEOMETRY
# ==============================================================================

def surrounding_grid_points(vlat, vlon):
    """Return the 4 surrounding 0.25-deg grid centres for a location, nearest-first."""
    base_lat = np.floor((vlat - GRID_OFFSET) / GRID_STEP) * GRID_STEP + GRID_OFFSET
    base_lon = np.floor((vlon - GRID_OFFSET) / GRID_STEP) * GRID_STEP + GRID_OFFSET
    pts = []
    for dlat in (0, GRID_STEP):
        for dlon in (0, GRID_STEP):
            glat = round(base_lat + dlat, 4)
            glon = round(base_lon + dlon, 4)
            dist = float(np.sqrt((vlat - glat) ** 2 + (vlon - glon) ** 2))
            pts.append((glat, glon, dist))
    pts.sort(key=lambda x: x[2])
    return pts


# ==============================================================================
# FILTERED PARQUET READ (predicate pushdown — unchanged core idea)
# ==============================================================================

def _read_parquet_grids(path, grid_pairs, columns=None):
    if not os.path.isfile(path):
        return None

    lats = sorted({round(float(la), 4) for la, _ in grid_pairs})
    lons = sorted({round(float(lo), 4) for _, lo in grid_pairs})

    try:
        import pyarrow.dataset as ds
        import pyarrow.compute as pc

        dataset = ds.dataset(path, format="parquet")
        flt = pc.field("lat").isin(lats) & pc.field("lon").isin(lons)
        table = dataset.to_table(filter=flt, columns=columns)
        df = table.to_pandas()
    except Exception:
        df = _read_parquet_grids_fallback(path, lats, lons, columns)
        if df is None:
            return None

    if df.empty:
        return df

    df["lat"] = df["lat"].round(4)
    df["lon"] = df["lon"].round(4)
    wanted = {(round(float(la), 4), round(float(lo), 4)) for la, lo in grid_pairs}
    mask = [(la, lo) in wanted for la, lo in zip(df["lat"], df["lon"])]
    return df.loc[mask].reset_index(drop=True)


def _read_parquet_grids_fallback(path, lats, lons, columns):
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        frames = []
        for batch in pf.iter_batches(batch_size=1_000_000, columns=columns):
            chunk = batch.to_pandas()
            chunk = chunk[chunk["lat"].round(4).isin(lats) &
                          chunk["lon"].round(4).isin(lons)]
            if not chunk.empty:
                frames.append(chunk)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    except Exception:
        df = pd.read_parquet(path, columns=columns)
        return df[df["lat"].round(4).isin(lats) & df["lon"].round(4).isin(lons)]


# ==============================================================================
# TRAINING DATA (filtered to nearby grids only)
# ==============================================================================

def build_grid_training(era5_dir, forecast_path, imd_dir, elev_path, var_key,
                        grid_pairs, verbose=False):
    elev_dict = {}
    if os.path.isfile(elev_path):
        elev_df = pd.read_parquet(elev_path)
        elev_dict = dict(zip(
            zip(elev_df["lat"].round(4), elev_df["lon"].round(4)),
            elev_df["elevation"],
        ))

    era5_file = os.path.join(era5_dir, f"training_{var_key}_daily.parquet")
    era5 = _read_parquet_grids(era5_file, grid_pairs,
                               columns=["lat", "lon", "date", "imd", "om"])
    if era5 is None or era5.empty:
        era5 = pd.DataFrame(columns=["lat", "lon", "date", "om", "imd", "source"])
    else:
        era5["date"] = pd.to_datetime(era5["date"]).dt.normalize()
        era5 = era5.dropna(subset=["imd", "om"])
        era5["source"] = 0
    _log(verbose, f"    [{var_key}] ERA5 rows: {len(era5):,}")

    fc = _read_parquet_grids(
        forecast_path, grid_pairs,
        columns=["lat", "lon", "date", var_key, "is_forecast"])
    fc_imd = pd.DataFrame(columns=["lat", "lon", "date", "om", "imd", "source"])
    if fc is not None and not fc.empty:
        fc["date"] = pd.to_datetime(fc["date"]).dt.normalize()
        fc["is_forecast"] = pd.to_numeric(fc["is_forecast"], errors="coerce")
        fc = fc[fc["is_forecast"] == 0]
        fc = fc.rename(columns={var_key: "om_fc"})
        fc = fc[["lat", "lon", "date", "om_fc"]].copy()
        fc["om_fc"] = pd.to_numeric(fc["om_fc"], errors="coerce")
        fc = fc.dropna(subset=["om_fc"])

        imd_file = os.path.join(imd_dir, f"imd_{var_key}_daily.parquet")
        imd = _read_parquet_grids(imd_file, grid_pairs,
                                  columns=["lat", "lon", "date", "value"])
        if imd is not None and not imd.empty:
            imd["date"] = pd.to_datetime(imd["date"]).dt.normalize()
            imd = imd.rename(columns={"value": "imd_fc"})
            imd["imd_fc"] = pd.to_numeric(imd["imd_fc"], errors="coerce")
            fc["lat"] = fc["lat"].round(4); fc["lon"] = fc["lon"].round(4)
            imd["lat"] = imd["lat"].round(4); imd["lon"] = imd["lon"].round(4)
            merged = fc.merge(imd[["lat", "lon", "date", "imd_fc"]],
                              on=["lat", "lon", "date"], how="inner")
            merged = merged.dropna(subset=["om_fc", "imd_fc"])
            merged = merged.rename(columns={"om_fc": "om", "imd_fc": "imd"})
            merged["source"] = 1
            fc_imd = merged[["lat", "lon", "date", "om", "imd", "source"]]
    _log(verbose, f"    [{var_key}] Forecast+IMD paired rows: {len(fc_imd):,}")

    combined = pd.concat(
        [era5[["lat", "lon", "date", "om", "imd", "source"]], fc_imd],
        ignore_index=True)
    if combined.empty:
        return {}
    combined = combined.sort_values(["lat", "lon", "date"]).reset_index(drop=True)
    combined["date"] = pd.to_datetime(combined["date"])
    combined["om"] = pd.to_numeric(combined["om"], errors="coerce")
    combined["imd"] = pd.to_numeric(combined["imd"], errors="coerce")
    combined = combined.dropna(subset=["om", "imd"]).reset_index(drop=True)

    doy = combined["date"].dt.dayofyear
    combined["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    combined["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    combined["month"] = combined["date"].dt.month
    combined["season_monsoon"]     = combined["month"].isin([6, 7, 8, 9]).astype(np.int8)
    combined["season_premonsoon"]  = combined["month"].isin([3, 4, 5]).astype(np.int8)
    combined["season_postmonsoon"] = combined["month"].isin([10, 11]).astype(np.int8)
    combined["season_winter"]      = combined["month"].isin([12, 1, 2]).astype(np.int8)

    combined["lat_r"] = combined["lat"].round(4)
    combined["lon_r"] = combined["lon"].round(4)
    combined["elevation"] = combined.apply(
        lambda r: elev_dict.get((r["lat_r"], r["lon_r"]), 0.0), axis=1)

    combined["om_roll7"] = (
        combined.groupby(["lat", "lon"])["om"]
        .transform(lambda x: x.rolling(7, min_periods=1).mean()))
    om_clim = combined.groupby(["lat", "lon", "month"])["om"].transform("mean")
    combined["om_clim_anom"] = combined["om"] - om_clim

    if var_key == "pcp":
        combined["rain_om"] = (combined["om"] >= RAIN_THRESHOLD).astype(np.int8)
        combined["om_log"] = np.log1p(combined["om"].clip(lower=0))

    train_cutoff = pd.Timestamp(TRAIN_END)
    grid_data = {}
    for (lat, lon), gdf in combined.groupby(["lat", "lon"]):
        train = gdf[gdf["date"] <= train_cutoff]
        grid_data[(round(lat, 4), round(lon, 4))] = train
    return grid_data


def train_grid_model(train, var_key):
    """
    Train a single grid's model in memory.
    Returns (model_data_dict_or_None, reason_str). reason_str is what tells
    you WHY a grid was skipped (missing xgboost, zero rows, too few rows,
    etc.) instead of every failure just looking like "insufficient data".
    """
    if not _XGB_AVAILABLE:
        return None, "xgboost is not installed in this environment"
    if train is None:
        return None, "no training rows at all for this grid (file not found or grid never matched)"
    if len(train) < MIN_TRAIN_ROWS:
        return None, f"only {len(train)} training rows (need >= {MIN_TRAIN_ROWS})"

    if var_key in ("tmax", "tmin"):
        feat = [f for f in TEMP_FEATURES if f in train.columns]
        model = xgb.XGBRegressor(**XGB_PARAMS, n_jobs=1)
        model.fit(train[feat].values, train["imd"].values)
        return {"type": "regressor", "model": model, "features": feat}, "ok"

    elif var_key == "pcp":
        cls_feat = [f for f in RAIN_CLS_FEATURES if f in train.columns]
        tc = train.copy()
        tc["rain_obs"] = (tc["imd"] >= RAIN_THRESHOLD).astype(int)
        y_cls = tc["rain_obs"].values
        if y_cls.sum() < MIN_RAIN_ROWS or (1 - y_cls.mean()) < 0.01:
            return None, (f"only {int(y_cls.sum())} rain-days out of {len(y_cls)} rows "
                          f"(need >= {MIN_RAIN_ROWS} and some dry days too)")

        cls_params = XGB_PARAMS.copy()
        cls_params["n_estimators"] = 100
        cls_params["max_depth"] = 4
        spw = (1 - y_cls.mean()) / max(y_cls.mean(), 1e-6)
        cls_model = xgb.XGBClassifier(**cls_params, scale_pos_weight=spw, n_jobs=1)
        cls_model.fit(tc[cls_feat].values, y_cls)

        rain_train = tc[tc["rain_obs"] == 1]
        reg_feat = [f for f in RAIN_REG_FEATURES if f in rain_train.columns]
        if len(rain_train) < MIN_RAIN_ROWS:
            return None, f"only {len(rain_train)} rain rows for the amount regressor (need >= {MIN_RAIN_ROWS})"
        reg_model = xgb.XGBRegressor(**XGB_PARAMS, n_jobs=1)
        reg_model.fit(rain_train[reg_feat].values,
                      np.log1p(rain_train["imd"].values))
        return {
            "type": "two_stage",
            "cls_model": cls_model, "cls_features": cls_feat,
            "reg_model": reg_model, "reg_features": reg_feat,
        }, "ok"
    return None, f"unrecognized var_key {var_key!r}"


# ==============================================================================
# MODEL CACHE  (disk-persisted, keyed by variable + grid point; shared across
# requests so nearby villages reuse each other's trained grids)
# ==============================================================================

_MEMORY_CACHE = {}  # (var_key, glat, glon) -> model_data ; lives for the process


def _cache_path(var_key, glat, glon):
    tag = f"{glat:.4f}_{glon:.4f}".replace("-", "m")
    return os.path.join(MODEL_CACHE_DIR, var_key, f"grid_{tag}.joblib")


def _cache_dump(obj, path):
    if _JOBLIB_AVAILABLE:
        joblib.dump(obj, path)
    else:
        with open(path, "wb") as f:
            pickle.dump(obj, f)


def _cache_load(path):
    if _JOBLIB_AVAILABLE:
        return joblib.load(path)
    with open(path, "rb") as f:
        return pickle.load(f)


def _get_cached_model(var_key, glat, glon):
    key = (var_key, glat, glon)
    today = date.today()
    if key in _MEMORY_CACHE:
        model_data, cache_date = _MEMORY_CACHE[key]
        if cache_date == today:
            return model_data
        else:
            del _MEMORY_CACHE[key]

    path = _cache_path(var_key, glat, glon)
    if not os.path.isfile(path):
        return None

    try:
        mtime = os.path.getmtime(path)
        file_date = datetime.fromtimestamp(mtime).date()
        if file_date != today:
            return None  # expired (prior to start of a new day)
    except Exception:
        return None

    try:
        model_data = _cache_load(path)
    except Exception:
        return None
    _MEMORY_CACHE[key] = (model_data, today)
    return model_data


def _set_cached_model(var_key, glat, glon, model_data):
    key = (var_key, glat, glon)
    _MEMORY_CACHE[key] = (model_data, date.today())
    try:
        path = _cache_path(var_key, glat, glon)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        _cache_dump(model_data, path)
    except Exception:
        pass  # cache is a performance optimization, never fatal


def clear_model_cache():
    """Clear both in-memory and disk caches."""
    _MEMORY_CACHE.clear()
    if os.path.exists(MODEL_CACHE_DIR):
        try:
            import shutil
            shutil.rmtree(MODEL_CACHE_DIR)
        except Exception:
            pass


def get_trained_grids(era5_dir, forecast_path, imd_dir, elev_path, var_key,
                      grid_pts, verbose=False):
    """
    Returns (trained_list, diagnostics) where trained_list is
    [(glat, glon, dist, model_data), ...] for grids with a usable model
    (pulling from cache wherever possible, training only what's missing or
    stale), and diagnostics is {(glat, glon): reason_str} covering every
    grid point, including the ones that succeeded ("ok" / "cache hit").
    """
    trained = {}
    diagnostics = {}
    to_train = []
    for glat, glon, dist in grid_pts:
        cached = _get_cached_model(var_key, glat, glon)
        if cached is not None:
            trained[(glat, glon)] = cached
            diagnostics[(glat, glon)] = "cache hit"
            _log(verbose, f"    [{var_key}] cache hit  ({glat}, {glon})")
        else:
            to_train.append((glat, glon, dist))

    if to_train:
        grid_pairs = [(g[0], g[1]) for g in to_train]
        grid_data = build_grid_training(
            era5_dir, forecast_path, imd_dir, elev_path, var_key, grid_pairs,
            verbose=verbose)
        for glat, glon, dist in to_train:
            md, reason = train_grid_model(grid_data.get((glat, glon)), var_key)
            if md is not None:
                _set_cached_model(var_key, glat, glon, md)
                trained[(glat, glon)] = md
                diagnostics[(glat, glon)] = "trained"
                _log(verbose, f"    [{var_key}] trained+cached ({glat}, {glon})")
            else:
                diagnostics[(glat, glon)] = reason
                _log(verbose, f"    [{var_key}] skipped ({glat}, {glon}) — {reason}")

    trained_list = [(glat, glon, dist, trained[(glat, glon)])
                    for glat, glon, dist in grid_pts if (glat, glon) in trained]
    return trained_list, diagnostics


# ==============================================================================
# ELEVATION + FORECAST FETCH  (paid -> free fallback)
# ==============================================================================

def fetch_elevation(lat, lon):
    try:
        resp = requests.get(ELEVATION_URL,
                            params={"locations": f"{lat:.6f},{lon:.6f}"},
                            timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == "OK" and data["results"]:
            elev = data["results"][0].get("elevation")
            return float(elev) if elev is not None else 0.0
        return 0.0
    except Exception:
        # Elevation is a helpful feature, not essential — degrade to 0 rather
        # than fail the whole request.
        return 0.0


def _parse_daily(data):
    if "daily" not in data:
        return None
    d = data["daily"]
    return pd.DataFrame({
        "date": pd.to_datetime(d["time"]),
        "tmax": pd.to_numeric(pd.Series(d.get("temperature_2m_max", [])), errors="coerce"),
        "tmin": pd.to_numeric(pd.Series(d.get("temperature_2m_min", [])), errors="coerce"),
        "pcp":  pd.to_numeric(pd.Series(d.get("precipitation_sum", [])), errors="coerce"),
    })


def parse_provided_forecast(forecast_data):
    """
    Parses raw forecast data passed from frontend/API caller into a pandas DataFrame.
    Supports:
      1. Dict with "daily" key: {"daily": {"time": [...], "temperature_2m_max": [...]}}
      2. Dict of arrays directly: {"time": [...], "temperature_2m_max": [...]}
      3. List of dicts (records): [{"time": "2026-07-06", "temperature_2m_max": 31.5}]
    """
    if isinstance(forecast_data, dict):
        if "daily" in forecast_data:
            d = forecast_data["daily"]
        else:
            d = forecast_data
        
        # Check if arrays are present
        time_arr = d.get("time") or d.get("date")
        tmax_arr = d.get("temperature_2m_max") or d.get("tmax")
        tmin_arr = d.get("temperature_2m_min") or d.get("tmin")
        pcp_arr = d.get("precipitation_sum") or d.get("pcp")
        
        if not time_arr:
            raise ValueError("Provided forecast_data must contain date/time information.")
        
        df = pd.DataFrame({
            "date": pd.to_datetime(time_arr),
            "tmax": pd.to_numeric(pd.Series(tmax_arr), errors="coerce") if tmax_arr is not None else np.nan,
            "tmin": pd.to_numeric(pd.Series(tmin_arr), errors="coerce") if tmin_arr is not None else np.nan,
            "pcp":  pd.to_numeric(pd.Series(pcp_arr), errors="coerce") if pcp_arr is not None else np.nan,
        })
    elif isinstance(forecast_data, list):
        # List of records
        df = pd.DataFrame(forecast_data)
        rename_map = {}
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in ("time", "date"):
                rename_map[col] = "date"
            elif col_lower in ("temperature_2m_max", "tmax"):
                rename_map[col] = "tmax"
            elif col_lower in ("temperature_2m_min", "tmin"):
                rename_map[col] = "tmin"
            elif col_lower in ("precipitation_sum", "pcp"):
                rename_map[col] = "pcp"
        df = df.rename(columns=rename_map)
        if "date" not in df.columns:
            raise ValueError("Provided forecast_data records must contain date or time field.")
        df["date"] = pd.to_datetime(df["date"])
        for col in ["tmax", "tmin", "pcp"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                df[col] = np.nan
    else:
        raise TypeError("forecast_data must be a dictionary or a list of records.")
    
    return df


def fetch_forecast(lat, lon, apikey, past_days=PAST_BUFFER_DAYS):
    """
    Returns (dataframe, source) where source is 'paid' or 'free'.
    Raises ForecastFetchError if no forecast could be retrieved at all.
    """
    use_free = False
    url = LIVE_FORECAST_URL
    df = None
    last_err = None

    for attempt in range(API_RETRY_ATTEMPTS):
        try:
            params = {
                "latitude": f"{lat:.6f}", "longitude": f"{lon:.6f}",
                "forecast_days": FORECAST_DAYS, "past_days": past_days,
                "daily": DAILY_PARAMS, "timezone": "auto",
            }
            if not use_free:
                params["apikey"] = apikey
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            df = _parse_daily(resp.json())
            break
        except requests.exceptions.HTTPError as e:
            last_err = e
            code = e.response.status_code if e.response is not None else None
            if not use_free and code in (400, 401, 402, 403):
                use_free = True
                url = FREE_FORECAST_URL
                continue
            if code == 429:
                time.sleep(10 * (attempt + 1))
                continue
            raise ForecastFetchError(f"Open-Meteo request failed with HTTP {code}") from e
        except requests.exceptions.RequestException as e:
            last_err = e
            time.sleep(5 * (attempt + 1))

    if df is None:
        raise ForecastFetchError(
            f"Could not fetch forecast after {API_RETRY_ATTEMPTS} attempts: {last_err}")

    df = df.drop_duplicates(subset=["date"], keep="last")
    df = df.sort_values("date").reset_index(drop=True)
    df["is_forecast"] = (df["date"] >= pd.Timestamp(date.today())).astype(int)
    return df, ("free" if use_free else "paid")


# ==============================================================================
# FEATURE ENGINEERING (forecast side)
# ==============================================================================

def add_features(df, elevation):
    df = df.copy()
    df["elevation"] = elevation
    df["source"] = 1
    doy = df["date"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["month"] = df["date"].dt.month
    df["season_monsoon"]     = df["month"].isin([6, 7, 8, 9]).astype(np.int8)
    df["season_premonsoon"]  = df["month"].isin([3, 4, 5]).astype(np.int8)
    df["season_postmonsoon"] = df["month"].isin([10, 11]).astype(np.int8)
    df["season_winter"]      = df["month"].isin([12, 1, 2]).astype(np.int8)
    return df


def add_var_features(df, var_col):
    df = df.copy()
    df["om"] = pd.to_numeric(df[var_col], errors="coerce")
    df["om_roll7"] = df["om"].rolling(7, min_periods=1).mean()
    om_clim = df.groupby("month")["om"].transform("mean")
    df["om_clim_anom"] = df["om"] - om_clim
    df["rain_om"] = (df["om"] >= RAIN_THRESHOLD).astype(np.int8)
    df["om_log"] = np.log1p(df["om"].clip(lower=0))
    return df


# ==============================================================================
# IDW PREDICTION FROM CACHED / IN-MEMORY MODELS
# ==============================================================================

def predict_idw(df_var, var_key, trained_models):
    if not trained_models:
        return df_var["om"].values

    predictions, weights = [], []
    for glat, glon, dist, md in trained_models:
        om = df_var["om"].values
        if var_key in ("tmax", "tmin"):
            feat = [f for f in md["features"] if f in df_var.columns]
            pred = md["model"].predict(df_var[feat].values)
        elif var_key == "pcp":
            cls_feat = [f for f in md["cls_features"] if f in df_var.columns]
            reg_feat = [f for f in md["reg_features"] if f in df_var.columns]
            cls_pred = md["cls_model"].predict(df_var[cls_feat].values)
            pred = np.copy(om)
            above = om >= RAIN_THRESHOLD
            pred[above & (cls_pred == 0)] = 0.0
            correct = above & (cls_pred == 1)
            if correct.sum() > 0:
                lp = md["reg_model"].predict(df_var[correct][reg_feat].values)
                pred[correct] = np.maximum(np.expm1(lp), RAIN_THRESHOLD)
        else:
            pred = om
        predictions.append(pred)
        weights.append(1.0 / max(dist, 0.001))

    weights = np.array(weights)
    weights = weights / weights.sum()
    result = np.zeros_like(predictions[0], dtype=float)
    for pred, w in zip(predictions, weights):
        result += w * pred
    return result


# ==============================================================================
# MAIN CALLABLE
# ==============================================================================

def get_village_forecast(lat, lon, *, verbose=False,
                         era5_dir=ERA5_DIR, forecast_path=FORECAST_PARQUET,
                         imd_dir=IMD_DIR, elev_path=ELEV_PARQUET,
                         apikey=OPEN_METEO_API_KEY, forecast_data=None):
    """
    Compute a bias-corrected 16-day forecast for a single lat/lon.

    Returns a JSON-serializable dict:

      success == True:
        {
          "success": True,
          "location": {"lat": .., "lon": .., "elevation_m": ..},
          "grids_used": [{"lat":.., "lon":.., "distance_deg":.., "cached": bool}, ...],
          "forecast_source": "paid" | "free" | "provided",
          "forecast": [
            {"date": "YYYY-MM-DD",
             "tmax_raw": .., "tmax_corrected": ..,
             "tmin_raw": .., "tmin_corrected": ..,
             "pcp_raw": .., "pcp_corrected": ..},
            ...
          ]
        }

      success == False:
        {"success": False, "error": {"type": "...", "message": "..."}}
    """
    try:
        lat, lon = _validate_location(lat, lon)

        grid_pts = surrounding_grid_points(lat, lon)
        _log(verbose, f"Grids: {grid_pts}")

        # Upfront sanity check: if the training data directories/files don't
        # even exist, every grid will silently look "insufficient" — surface
        # that clearly instead of letting it masquerade as a data problem.
        path_check = {
            "era5_dir": era5_dir,
            "forecast_parquet": forecast_path,
            "imd_dir": imd_dir,
            "elevation_parquet": elev_path,
        }
        missing_paths = {k: v for k, v in path_check.items()
                         if not os.path.exists(v)}
        if verbose and missing_paths:
            _log(verbose, f"WARNING: these configured paths do not exist: {missing_paths}")

        # Track which grids were already cached vs freshly trained, per-variable,
        # for transparency in the response.
        grids_cache_status = {(g[0], g[1]): (_get_cached_model(v, g[0], g[1]) is not None)
                              for g in grid_pts for v in VAR_KEYS}

        elevation = fetch_elevation(lat, lon)
        _log(verbose, f"Elevation: {elevation}")

        if forecast_data is not None:
            forecast = parse_provided_forecast(forecast_data)
            forecast = forecast.drop_duplicates(subset=["date"], keep="last")
            forecast = forecast.sort_values("date").reset_index(drop=True)
            forecast["is_forecast"] = (forecast["date"] >= pd.Timestamp(date.today())).astype(int)
            forecast_source = "provided"
        else:
            forecast, forecast_source = fetch_forecast(lat, lon, apikey)

        forecast = add_features(forecast, elevation)

        results = forecast[["date", "is_forecast"]].copy()
        variable_status = {}

        for var_key, var_col in [("tmax", "tmax"), ("tmin", "tmin"), ("pcp", "pcp")]:
            trained, grid_diagnostics = get_trained_grids(
                era5_dir, forecast_path, imd_dir, elev_path, var_key, grid_pts,
                verbose=verbose)

            df_var = add_var_features(forecast, var_col)

            grid_reasons = [
                {"lat": g[0], "lon": g[1], "reason": grid_diagnostics.get((g[0], g[1]), "unknown")}
                for g in grid_pts
            ]

            if not trained:
                variable_status[var_key] = {
                    "status": "raw_fallback",
                    "reason": "no grid had a usable model — see grid_diagnostics",
                    "grid_diagnostics": grid_reasons,
                }
                if missing_paths:
                    variable_status[var_key]["missing_paths"] = missing_paths
                results[f"{var_key}_raw"] = df_var["om"].values
                results[f"{var_key}_corrected"] = df_var["om"].values
                continue

            corrected = predict_idw(df_var, var_key, trained)
            variable_status[var_key] = {
                "status": "corrected",
                "grids_trained": len(trained),
                "grid_diagnostics": grid_reasons,
            }
            results[f"{var_key}_raw"] = df_var["om"].values
            results[f"{var_key}_corrected"] = corrected

        # Keep only true forecast days (drop the rolling-feature lead-in buffer)
        results = results[results["is_forecast"] == 1].reset_index(drop=True)

        forecast_rows = []
        for _, row in results.iterrows():
            entry = {"date": row["date"].strftime("%Y-%m-%d")}
            for var_key in VAR_KEYS:
                for suffix in ("raw", "corrected"):
                    col = f"{var_key}_{suffix}"
                    if col in results.columns:
                        val = row[col]
                        entry[col] = None if pd.isna(val) else round(float(val), 2)
            forecast_rows.append(entry)

        grids_used = [
            {
                "lat": g[0], "lon": g[1], "distance_deg": round(g[2], 4),
                "cached": grids_cache_status.get((g[0], g[1]), False),
            }
            for g in grid_pts
        ]

        return {
            "success": True,
            "location": {"lat": lat, "lon": lon, "elevation_m": round(elevation, 1)},
            "grids_used": grids_used,
            "forecast_source": forecast_source,
            "forecast": forecast_rows,
        }

    except VillageForecastError as e:
        return {"success": False, "error": {"type": e.error_type, "message": str(e)}}
    except Exception as e:
        # Catch-all so the caller always gets a well-formed response, never a
        # raised exception.
        return {"success": False, "error": {"type": "internal_error", "message": str(e)}}


# ==============================================================================
# Manual test entry point (not required for library use)
# ==============================================================================

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python village_forecast_service.py <lat> <lon>")
        sys.exit(1)
    out = get_village_forecast(float(sys.argv[1]), float(sys.argv[2]), verbose=True)
    print(json.dumps(out, indent=2))