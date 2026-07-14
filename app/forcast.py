"""
Village Forecast — Combined Per-Grid Train + Forecast (light on render)
=======================================================================
Single-shot pipeline that replaces the two-step workflow:

    pergrid_train.py            -> ./models_pergrid/{var}/grid_*.joblib
    village_forecast_pergrid.py -> reads ./models_pergrid, forecasts

Instead of training ALL grids and saving them to disk, this script:

  1. Takes a single location from CLI (--lat / --lon)
  2. Finds the 4 surrounding 0.25 deg grid points
  3. Loads ONLY those 4 grids' rows from the (huge) parquets, using a
     pyarrow predicate-pushdown filter so ~58M-row files are never fully
     read into memory  ("light on render")
  4. Trains the 4 per-grid models IN MEMORY (nothing written to ./models_pergrid)
  5. Fetches the Open-Meteo forecast for the location, with the same
     paid -> free API fallback on an expired / invalid / over-quota key
  6. Applies inverse-distance-weighted (IDW) correction from the 4 models
  7. Writes the corrected village forecast CSV (+ optional parquet)

Because models are never saved and only 4 grids are ever touched, memory
and CPU stay tiny regardless of how large the source parquets are.

Usage
-----
    python village_forecast_combined.py \
        --lat 22.3 --lon 72.6 \
        --era5_dir ./training_data \
        --forecast ./forecast_data/om_forecast_all.parquet \
        --imd_dir ./imd_processed \
        --elev ./ml_ready/grid_elevation.parquet \
        --apikey XLv82nM2BPBe6qVe \
        --output_dir ./out

The --forecast parquet is only used for TRAINING (past forecast-vs-IMD
pairs). The live forecast for the target location is fetched from
Open-Meteo at run time.
"""

import os
import sys
import time
import argparse
import warnings
import numpy as np
import pandas as pd
import requests
from datetime import date, timedelta
import xgboost as xgb

warnings.filterwarnings("ignore")


# -- Config --------------------------------------------------------------------

RAIN_THRESHOLD = 1.0
TRAIN_END = "2024-12-31"        # train on data up to end of 2024
TEST_START = "2025-01-01"       # hold out 2025+ for skill evaluation
GRID_STEP = 0.25
GRID_OFFSET = 0.125

OUTPUT_PAST_DAYS = 85           # past days to include in the corrected output
ROLL_BUFFER = 7                 # extra past days fetched only to warm up om_roll7
MAX_FORECAST_FILES = 100        # maximum number of forecast files to keep in output_dir

HIST_FORECAST_URL = "https://customer-historical-forecast-api.open-meteo.com/v1/forecast"
LIVE_FORECAST_URL = "https://customer-api.open-meteo.com/v1/forecast"
FREE_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
FREE_ARCHIVE_URL  = "https://archive-api.open-meteo.com/v1/archive"
ELEVATION_URL     = "https://api.opentopodata.org/v1/srtm90m"
DAILY_PARAMS      = "temperature_2m_max,temperature_2m_min,precipitation_sum"

# Lighter XGBoost params for per-grid models (~14k samples each)
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

# Features (no lat/lon — each model is grid-specific)
TEMP_FEATURES = [
    "om", "elevation",
    "doy_sin", "doy_cos", "month",
    "season_monsoon", "season_premonsoon", "season_postmonsoon", "season_winter",
    "om_roll7", "om_clim_anom",
    "source",  # 0=ERA5, 1=forecast
]
RAIN_CLS_FEATURES = TEMP_FEATURES + ["rain_om"]
RAIN_REG_FEATURES = ["om_log"] + TEMP_FEATURES


# ==============================================================================
# GRID GEOMETRY
# ==============================================================================

def surrounding_grid_points(vlat, vlon):
    """Return the 4 surrounding 0.25-deg grid centres for a location."""
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
    return pts  # list of (glat, glon, dist) nearest-first


# ==============================================================================
# FILTERED PARQUET READ  (the "light on render" core)
# ==============================================================================

def _read_parquet_grids(path, grid_pairs, columns=None):
    """
    Read only the rows matching the requested (lat, lon) grid pairs.

    Uses pyarrow predicate pushdown when available so that a ~58M-row parquet
    is never fully materialised. Falls back to a chunked pandas scan if
    pyarrow filtering is unavailable.

    grid_pairs : iterable of (lat, lon) floats (already rounded to grid centres)
    """
    if not os.path.isfile(path):
        return None

    lats = sorted({round(float(la), 4) for la, _ in grid_pairs})
    lons = sorted({round(float(lo), 4) for _, lo in grid_pairs})
    # Bounding box with a small pad so float32/float64 storage drift can't
    # exclude a grid centre that is nominally in range.
    pad = 0.01
    lat_lo, lat_hi = min(lats) - pad, max(lats) + pad
    lon_lo, lon_hi = min(lons) - pad, max(lons) + pad

    # --- Preferred path: pyarrow dataset with a pushdown RANGE filter ---
    try:
        import pyarrow.dataset as ds
        import pyarrow.compute as pc

        dataset = ds.dataset(path, format="parquet")
        flt = ((pc.field("lat") >= lat_lo) & (pc.field("lat") <= lat_hi) &
               (pc.field("lon") >= lon_lo) & (pc.field("lon") <= lon_hi))
        table = dataset.to_table(filter=flt, columns=columns)
        df = table.to_pandas()
    except Exception:
        # --- Fallback: chunked read with range filtering ---
        df = _read_parquet_grids_fallback(path, lat_lo, lat_hi, lon_lo, lon_hi,
                                          columns)
        if df is None:
            return None

    if df.empty:
        return df

    # Snap each row to the nearest requested grid centre; keep only rows within
    # half a grid step of a requested centre (handles precision drift and any
    # sub-grid offset between the OM / IMD / ERA products).
    tol = GRID_STEP / 2.0 + 1e-6
    df = df.copy()
    lat_arr = df["lat"].to_numpy(dtype=float)
    lon_arr = df["lon"].to_numpy(dtype=float)
    lats_np = np.array(lats)
    lons_np = np.array(lons)
    nearest_lat = lats_np[np.abs(lat_arr[:, None] - lats_np[None, :]).argmin(axis=1)]
    nearest_lon = lons_np[np.abs(lon_arr[:, None] - lons_np[None, :]).argmin(axis=1)]
    keep = (np.abs(lat_arr - nearest_lat) <= tol) & \
           (np.abs(lon_arr - nearest_lon) <= tol)
    df["lat"] = np.round(nearest_lat, 4)
    df["lon"] = np.round(nearest_lon, 4)
    df = df.loc[keep].reset_index(drop=True)

    # Final refine to the exact requested (lat, lon) pairs.
    wanted = {(round(float(la), 4), round(float(lo), 4)) for la, lo in grid_pairs}
    mask = [(la, lo) in wanted for la, lo in zip(df["lat"], df["lon"])]
    return df.loc[mask].reset_index(drop=True)


def _read_parquet_grids_fallback(path, lat_lo, lat_hi, lon_lo, lon_hi, columns):
    """Chunked fallback when pyarrow dataset filtering is unavailable."""
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        frames = []
        for batch in pf.iter_batches(batch_size=1_000_000, columns=columns):
            chunk = batch.to_pandas()
            chunk = chunk[(chunk["lat"] >= lat_lo) & (chunk["lat"] <= lat_hi) &
                          (chunk["lon"] >= lon_lo) & (chunk["lon"] <= lon_hi)]
            if not chunk.empty:
                frames.append(chunk)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    except Exception:
        # Last resort: full pandas read (heavy, but correct)
        df = pd.read_parquet(path, columns=columns)
        return df[(df["lat"] >= lat_lo) & (df["lat"] <= lat_hi) &
                  (df["lon"] >= lon_lo) & (df["lon"] <= lon_hi)]


# ==============================================================================
# TRAINING DATA (filtered to nearby grids only)
# ==============================================================================

def build_grid_training(era5_dir, forecast_path, imd_dir, elev_path,
                        var_key, grid_pairs, train_source="om"):
    """
    Build per-grid training frames for ONLY the requested grid points.
    Returns {(lat, lon): {"train", "test"}}.

    train_source controls which sources feed the model:
      "om"   -> OM forecast-history vs IMD only (source-matched to apply target)
      "era5" -> ERA5 vs IMD only
      "both" -> ERA5 + OM (original behaviour; can bias low-error vars like tmin)
    Note: the 2025+ TEST split always uses OM rows, since OM is the apply target.
    """
    # -- Elevation (small file, read whole then filter) --
    elev_dict = {}
    if os.path.isfile(elev_path):
        elev_df = pd.read_parquet(elev_path)
        elev_dict = dict(zip(
            zip(elev_df["lat"].round(4), elev_df["lon"].round(4)),
            elev_df["elevation"],
        ))

    # -- ERA5 training (filtered) — only if requested --
    era5 = pd.DataFrame(columns=["lat", "lon", "date", "om", "imd", "source"])
    if train_source in ("era5", "both"):
        era5_file = os.path.join(era5_dir, f"training_{var_key}_daily.parquet")
        era5 = _read_parquet_grids(era5_file, grid_pairs,
                                   columns=["lat", "lon", "date", "imd", "om"])
        if era5 is None or era5.empty:
            era5 = pd.DataFrame(columns=["lat", "lon", "date", "om", "imd", "source"])
        else:
            era5["date"] = pd.to_datetime(era5["date"]).dt.normalize()
            era5 = era5.dropna(subset=["imd", "om"])
            era5["source"] = 0

    # -- Forecast history (filtered), paired with IMD --
    fc = _read_parquet_grids(
        forecast_path, grid_pairs,
        columns=["lat", "lon", "date", var_key, "is_forecast"])
    fc_imd = pd.DataFrame(columns=["lat", "lon", "date", "om", "imd", "source"])
    if fc is not None and not fc.empty:
        fc["date"] = pd.to_datetime(fc["date"]).dt.normalize()
        fc["is_forecast"] = pd.to_numeric(fc["is_forecast"], errors="coerce")
        n_fc_all = len(fc)
        fc = fc[fc["is_forecast"] == 0]
        n_fc_past = len(fc)
        fc = fc.rename(columns={var_key: "om_fc"})
        fc = fc[["lat", "lon", "date", "om_fc"]].copy()
        fc["om_fc"] = pd.to_numeric(fc["om_fc"], errors="coerce")
        fc = fc.dropna(subset=["om_fc"])
        n_fc_valid = len(fc)

        imd_file = os.path.join(imd_dir, f"imd_{var_key}_daily.parquet")
        imd = _read_parquet_grids(imd_file, grid_pairs,
                                  columns=["lat", "lon", "date", "value"])
        n_imd = 0 if (imd is None or imd.empty) else len(imd)
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

    # -- Combine --
    combined = pd.concat(
        [era5[["lat", "lon", "date", "om", "imd", "source"]], fc_imd],
        ignore_index=True)
    if combined.empty:
        return {}
    combined = combined.sort_values(["lat", "lon", "date"]).reset_index(drop=True)
    combined["date"] = pd.to_datetime(combined["date"])
    # Force numeric dtype — filtered/concat reads can yield object-dtype columns
    combined["om"] = pd.to_numeric(combined["om"], errors="coerce")
    combined["imd"] = pd.to_numeric(combined["imd"], errors="coerce")
    combined = combined.dropna(subset=["om", "imd"]).reset_index(drop=True)

    # -- Features (identical to pergrid_train.py) --
    doy = combined["date"].dt.dayofyear
    combined["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    combined["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    combined["month"] = combined["date"].dt.month
    combined["season_monsoon"]    = combined["month"].isin([6, 7, 8, 9]).astype(np.int8)
    combined["season_premonsoon"] = combined["month"].isin([3, 4, 5]).astype(np.int8)
    combined["season_postmonsoon"]= combined["month"].isin([10, 11]).astype(np.int8)
    combined["season_winter"]     = combined["month"].isin([12, 1, 2]).astype(np.int8)

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

    # -- Split into per-grid dict: train (<=2024) and test (2025+) --
    # TRAIN uses whatever sources were loaded (train_source). TEST is always
    # OM-only (source==1), because OM is what we correct at apply time — this
    # keeps the skill number honest regardless of train_source.
    train_cutoff = pd.Timestamp(TRAIN_END)
    test_cutoff = pd.Timestamp(TEST_START)
    grid_data = {}
    for (lat, lon), gdf in combined.groupby(["lat", "lon"]):
        train = gdf[gdf["date"] <= train_cutoff]
        test = gdf[(gdf["date"] >= test_cutoff) & (gdf["source"] == 1)]
        grid_data[(round(lat, 4), round(lon, 4))] = {"train": train, "test": test}
    return grid_data


# ==============================================================================
# IN-MEMORY MODEL TRAINING  (no disk writes)
# ==============================================================================

def _eval_temp(model, feat, test):
    """Return (raw_mae, corr_mae, n) on the 2025+ test split for temperature."""
    if test is None or len(test) == 0:
        return None, None, 0
    tv = test.dropna(subset=["imd"])
    if len(tv) == 0:
        return None, None, 0
    pred = model.predict(tv[feat].values)
    raw = float(np.abs(tv["om"].values - tv["imd"].values).mean())
    corr = float(np.abs(pred - tv["imd"].values).mean())
    return raw, corr, len(tv)


def _eval_pcp(cls_model, cls_feat, reg_model, reg_feat, test):
    """Return (raw_mae, corr_mae, n) on the 2025+ test split for precipitation."""
    if test is None or len(test) == 0:
        return None, None, 0
    tv = test.dropna(subset=["imd"])
    if len(tv) == 0:
        return None, None, 0
    om_t = tv["om"].values
    imd_t = tv["imd"].values
    raw = float(np.abs(om_t - imd_t).mean())
    cls_pred = cls_model.predict(tv[cls_feat].values)
    pred = np.copy(om_t)
    above = om_t >= RAIN_THRESHOLD
    pred[above & (cls_pred == 0)] = 0.0
    correct = above & (cls_pred == 1)
    if correct.sum() > 0:
        lp = reg_model.predict(tv[correct][reg_feat].values)
        pred[correct] = np.maximum(np.expm1(lp), RAIN_THRESHOLD)
    corr = float(np.abs(pred - imd_t).mean())
    return raw, corr, len(tv)


def train_grid_model(train, test, var_key):
    """
    Train a single grid's model on <=2024 data and evaluate on 2025+ test.
    Returns a model-data dict (with raw_mae/corr_mae/test_n) or None.
    """
    if train is None or len(train) < 100:
        return None

    if var_key in ("tmax", "tmin"):
        feat = [f for f in TEMP_FEATURES if f in train.columns]
        model = xgb.XGBRegressor(**XGB_PARAMS, n_jobs=1)
        model.fit(train[feat].values, train["imd"].values)
        raw, corr, n = _eval_temp(model, feat, test)
        return {"type": "regressor", "model": model, "features": feat,
                "raw_mae": raw, "corr_mae": corr, "test_n": n}

    elif var_key == "pcp":
        cls_feat = [f for f in RAIN_CLS_FEATURES if f in train.columns]
        tc = train.copy()
        tc["rain_obs"] = (tc["imd"] >= RAIN_THRESHOLD).astype(int)
        y_cls = tc["rain_obs"].values
        if y_cls.sum() < 20 or (1 - y_cls.mean()) < 0.01:
            return None

        cls_params = XGB_PARAMS.copy()
        cls_params["n_estimators"] = 100
        cls_params["max_depth"] = 4
        spw = (1 - y_cls.mean()) / max(y_cls.mean(), 1e-6)
        cls_model = xgb.XGBClassifier(**cls_params, scale_pos_weight=spw, n_jobs=1)
        cls_model.fit(tc[cls_feat].values, y_cls)

        rain_train = tc[tc["rain_obs"] == 1]
        reg_feat = [f for f in RAIN_REG_FEATURES if f in rain_train.columns]
        if len(rain_train) < 20:
            return None
        reg_model = xgb.XGBRegressor(**XGB_PARAMS, n_jobs=1)
        reg_model.fit(rain_train[reg_feat].values,
                      np.log1p(rain_train["imd"].values))
        raw, corr, n = _eval_pcp(cls_model, cls_feat, reg_model, reg_feat, test)
        return {
            "type": "two_stage",
            "cls_model": cls_model, "cls_features": cls_feat,
            "reg_model": reg_model, "reg_features": reg_feat,
            "raw_mae": raw, "corr_mae": corr, "test_n": n,
        }
    return None


def train_surrounding_models(era5_dir, forecast_path, imd_dir, elev_path,
                             var_key, grid_pts, train_source="om"):
    """
    Train models for the surrounding grids (in memory).
    Returns list of (glat, glon, dist, model_data) for grids that trained OK.
    """
    grid_pairs = [(g[0], g[1]) for g in grid_pts]
    grid_data = build_grid_training(
        era5_dir, forecast_path, imd_dir, elev_path, var_key, grid_pairs,
        train_source=train_source)

    trained = []
    for glat, glon, dist in grid_pts:
        gd = grid_data.get((glat, glon))
        if gd is None:
            print(f"    skipped grid ({glat}, {glon}) — no data")
            continue
        md = train_grid_model(gd["train"], gd["test"], var_key)
        if md is not None:
            trained.append((glat, glon, dist, md))
    return trained


# ==============================================================================
# ELEVATION + FORECAST FETCH  (with paid -> free fallback)
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
    except Exception:
        pass
    return 0.0


def _parse_daily(data):
    if "daily" not in data:
        return None
    d = data["daily"]
    return pd.DataFrame({
        "date": pd.to_datetime(d["time"]).normalize(),
        "tmax": pd.to_numeric(pd.Series(d.get("temperature_2m_max", [])), errors="coerce"),
        "tmin": pd.to_numeric(pd.Series(d.get("temperature_2m_min", [])), errors="coerce"),
        "pcp":  pd.to_numeric(pd.Series(d.get("precipitation_sum", [])), errors="coerce"),
    })


def fetch_forecast(lat, lon, apikey, past_days=OUTPUT_PAST_DAYS + ROLL_BUFFER):
    """
    Fetch past_days of recent history + the 16-day live forecast from the
    Open-Meteo forecast API (no historical-forecast leg — that product isn't
    part of the standard package). On an expired / invalid / over-quota key
    (HTTP 400/401/402/403) the paid endpoint falls back to the free API.

    `past_days` includes both the days we want in the output (OUTPUT_PAST_DAYS)
    and a small ROLL_BUFFER used only to warm up the 7-day rolling feature;
    the buffer days are dropped before output.
    """
    live_use_free = False
    live_url = LIVE_FORECAST_URL
    live_df = None

    for attempt in range(5):
        try:
            tz = "Asia/Kolkata" if (6.0 <= lat <= 38.0 and 68.0 <= lon <= 98.0) else "GMT"
            params = {
                "latitude": f"{lat:.6f}", "longitude": f"{lon:.6f}",
                "forecast_days": 16, "past_days": past_days,
                "daily": DAILY_PARAMS, "timezone": tz,
            }
            if not live_use_free:
                params["apikey"] = apikey
            resp = requests.get(live_url, params=params, timeout=60)
            resp.raise_for_status()
            live_df = _parse_daily(resp.json())
            break
        except requests.exceptions.HTTPError as e:
            code = e.response.status_code if e.response is not None else None
            if not live_use_free and code in (400, 401, 402, 403):
                live_use_free = True
                live_url = FREE_FORECAST_URL
                continue
            if code == 429:
                time.sleep(10 * (attempt + 1))
            else:
                raise
        except Exception:
            time.sleep(5 * (attempt + 1))

    if live_df is None:
        return None
    live_df = live_df.drop_duplicates(subset=["date"], keep="last")
    live_df = live_df.sort_values("date").reset_index(drop=True)
    live_df["is_forecast"] = (live_df["date"] >= pd.Timestamp(date.today())).astype(int)
    return live_df


# ==============================================================================
# FEATURE ENGINEERING (forecast side)
# ==============================================================================

def add_features(df, elevation):
    df = df.copy()
    df["elevation"] = elevation
    df["source"] = 1  # forecast
    doy = df["date"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["month"] = df["date"].dt.month
    df["season_monsoon"]    = df["month"].isin([6, 7, 8, 9]).astype(np.int8)
    df["season_premonsoon"] = df["month"].isin([3, 4, 5]).astype(np.int8)
    df["season_postmonsoon"]= df["month"].isin([10, 11]).astype(np.int8)
    df["season_winter"]     = df["month"].isin([12, 1, 2]).astype(np.int8)
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
# IDW PREDICTION FROM IN-MEMORY MODELS
# ==============================================================================

def predict_idw(df_var, var_key, trained_models):
    """
    trained_models : list of (glat, glon, dist, model_data)
    Applies each grid's model then inverse-distance-weights the results.
    """
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
# NEAREST IMD (filtered read)
# ==============================================================================

def find_nearest_imd(vlat, vlon, imd_dir):
    grid_lat = round((vlat - GRID_OFFSET) / GRID_STEP) * GRID_STEP + GRID_OFFSET
    grid_lon = round((vlon - GRID_OFFSET) / GRID_STEP) * GRID_STEP + GRID_OFFSET
    grid_lat = round(grid_lat, 4)
    grid_lon = round(grid_lon, 4)

    imd_data = {}
    for var_key, fname in [("pcp", "imd_pcp_daily.parquet"),
                           ("tmax", "imd_tmax_daily.parquet"),
                           ("tmin", "imd_tmin_daily.parquet")]:
        path = os.path.join(imd_dir, fname)
        df = _read_parquet_grids(path, [(grid_lat, grid_lon)],
                                 columns=["lat", "lon", "date", "value"])
        if df is None or df.empty:
            continue
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        nearest = df[["date", "value"]].rename(columns={"value": f"imd_{var_key}"})
        imd_data[var_key] = nearest
    return imd_data


# ==============================================================================
# MAIN LOCATION PIPELINE
# ==============================================================================

def process_location(
    lat,
    lon,
    era5_dir="./data/training_data",
    forecast_path="./data/om_forecast_all.parquet",
    imd_dir="./data/IMD_parquets",
    elev="./data/grid_elevation.parquet",
    apikey="xxx",
    train_source="om",
    output_dir="./out"
):
    grid_pts = surrounding_grid_points(lat, lon)
    elevation = fetch_elevation(lat, lon)
    forecast = fetch_forecast(lat, lon, apikey)
    if forecast is None:
        print("    ERROR: No forecast data!")
        return None
    forecast = add_features(forecast, elevation)
    results = forecast[["date", "is_forecast"]].copy()

    for var_key, var_col in [("tmax", "tmax"), ("tmin", "tmin"), ("pcp", "pcp")]:
        trained = train_surrounding_models(
            era5_dir, forecast_path, imd_dir, elev,
            var_key, grid_pts, train_source=train_source)

        df_var = add_var_features(forecast, var_col)
        if not trained:
            results[f"{var_key}_forecast_raw"] = df_var["om"].values
            results[f"{var_key}_forecast_corrected"] = df_var["om"].values
            continue

        corrected = predict_idw(df_var, var_key, trained)
        results[f"{var_key}_forecast_raw"] = df_var["om"].values
        results[f"{var_key}_forecast_corrected"] = corrected

        # IDW-weighted 2025+ test skill across the surrounding grids
        w, raw_acc, corr_acc = 0.0, 0.0, 0.0
        for _, _, dist, md in trained:
            if md.get("raw_mae") is not None and md.get("corr_mae") is not None:
                wi = 1.0 / max(dist, 0.001)
                raw_acc += wi * md["raw_mae"]
                corr_acc += wi * md["corr_mae"]
                w += wi
        if w > 0:
            raw_s, corr_s = raw_acc / w, corr_acc / w
            imp = (raw_s - corr_s) / raw_s * 100 if raw_s else 0.0

    # Keep the last OUTPUT_PAST_DAYS of history + all forecast rows;
    # drop only the oldest ROLL_BUFFER days used to warm up om_roll7.
    cutoff = pd.Timestamp(date.today()) - pd.Timedelta(days=OUTPUT_PAST_DAYS)
    results = results[results["date"] >= cutoff].reset_index(drop=True)

    # Metadata
    results["lat"] = lat
    results["lon"] = lon
    results["elevation"] = elevation

    # Merge nearest IMD observations (available for the past-days portion)
    if imd_dir and os.path.isdir(imd_dir):
        imd_data = find_nearest_imd(lat, lon, imd_dir)
        for _, imd_df in imd_data.items():
            results = results.merge(imd_df, on="date", how="left")

    col_order = [
        "lat", "lon", "elevation", "date", "is_forecast",
        "tmax_forecast_raw", "tmax_forecast_corrected", "imd_tmax",
        "tmin_forecast_raw", "tmin_forecast_corrected", "imd_tmin",
        "pcp_forecast_raw", "pcp_forecast_corrected", "imd_pcp",
    ]
    results = results[[c for c in col_order if c in results.columns]]

    os.makedirs(output_dir, exist_ok=True)
    tag = f"{lat:.4f}_{lon:.4f}".replace("-", "m")
    csv_path = os.path.join(output_dir, f"forecast_{tag}.csv")
    results.to_csv(csv_path, index=False, float_format="%.2f")

    # Keep only up to MAX_FORECAST_FILES in output_dir (act like a queue)
    import glob
    files = glob.glob(os.path.join(output_dir, "forecast_*.csv"))
    files.sort(key=os.path.getmtime)
    if len(files) > MAX_FORECAST_FILES:
        for f in files[:-MAX_FORECAST_FILES]:
            os.remove(f)

    # Live-window check vs IMD (past-days portion). NOTE: this uses short-window
    # features (om_roll7 / om_clim_anom computed over the fetched window only),
    # so it is a rough diagnostic, NOT the model's true skill. The trustworthy
    # number is the "2025+ skill (IDW)" line printed during training above.
    past = results[results["is_forecast"] == 0]
    printed_header = False
    for var_key in ["tmax", "tmin", "pcp"]:
        imd_col = f"imd_{var_key}"
        if imd_col in past.columns:
            valid = past[past[imd_col].notna()]
            if len(valid) > 1:
                if not printed_header:
                    printed_header = True

    # Filter to 16-day forecast and format for JSON return
    forecast_16 = results[results["is_forecast"] == 1].copy()
    forecast_16 = forecast_16[["date", "tmax_forecast_corrected", "tmin_forecast_corrected", "pcp_forecast_corrected"]]
    forecast_16.columns = ["date", "tmax", "tmin", "pcp"]
    
    # Format date as YYYY-MM-DD
    if pd.api.types.is_datetime64_any_dtype(forecast_16["date"]):
        forecast_16["date"] = forecast_16["date"].dt.strftime("%Y-%m-%d")
    else:
        forecast_16["date"] = pd.to_datetime(forecast_16["date"]).dt.strftime("%Y-%m-%d")
        
    return forecast_16.to_json(orient="records")


def run_forecast_pipeline(
    lat,
    lon,
    era5_dir="./data/training_data",
    forecast="./data/om_forecast_all.parquet",
    imd_dir="./data/IMD_parquets",
    elev="./data/grid_elevation.parquet",
    apikey="xxx",
    train_source="om",
    output_dir="./out"
):
    import time
    t0 = time.time()
    res = process_location(
        lat=lat,
        lon=lon,
        era5_dir=era5_dir,
        forecast_path=forecast,
        imd_dir=imd_dir,
        elev=elev,
        apikey=apikey,
        train_source=train_source,
        output_dir=output_dir
    )
    duration = time.time() - t0
    try:
        from app.stats import update_stats
        update_stats("forecast", duration)
    except Exception as e:
        print(f"[STATS WARNING] Error logging stats: {e}")
    return res


# ==============================================================================
# CLI
# ==============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Combined per-grid train + village forecast (light on render)")
    p.add_argument("--lat", type=float, required=True, help="Target latitude")
    p.add_argument("--lon", type=float, required=True, help="Target longitude")
    p.add_argument("--era5_dir", default="./data/training_data",
                   help="Dir with training_*_daily.parquet")
    p.add_argument("--forecast", default="./data/om_forecast_all.parquet",
                   help="Forecast parquet (used for training pairs only)")
    p.add_argument("--imd_dir", default="./data/IMD_parquets",
                   help="Dir with imd_*_daily.parquet")
    p.add_argument("--elev", default="./data/grid_elevation.parquet",
                   help="Grid elevation parquet")
    p.add_argument("--apikey", required=False, default="xxx", help="Open-Meteo customer API key")
    p.add_argument("--train_source", choices=["om", "era5", "both"],
                   default="om",
                   help="Which sources train the model. 'om' (default) is "
                        "source-matched to the OM apply target; 'both' adds "
                        "ERA5 (can bias low-error vars like tmin).")
    p.add_argument("--output_dir", default="./out", help="Output CSV directory")
    args = p.parse_args()

    res_json = run_forecast_pipeline(
        lat=args.lat,
        lon=args.lon,
        era5_dir=args.era5_dir,
        forecast=args.forecast,
        imd_dir=args.imd_dir,
        elev=args.elev,
        apikey=args.apikey,
        train_source=args.train_source,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
