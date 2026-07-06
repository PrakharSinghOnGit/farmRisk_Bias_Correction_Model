"""
Village Soil Moisture Service — Checkpointed / Incremental
=============================================================
Efficient replacement for run_village_soil_moisture.py.

THE CORE PROBLEM WITH THE ORIGINAL
-----------------------------------
The CPC leaky-bucket water balance is a Markov recursion:

    w(t+1) = f(w(t), snowpack(t), forcing(t+1))

Day N+1 only ever needs day N's state. The original script re-derives that
state by replaying the ENTIRE historical record (often 30+ years, via a
plain Python day-by-day loop) on every single call, just to get to
yesterday, so it can compute one new day. That's the actual bottleneck --
not the model's math, which is cheap per day.

WHAT THIS FILE DOES INSTEAD
----------------------------
Per village, it persists a small CHECKPOINT to disk after every run:
    - last integrated date
    - w (storage) and snowpack at that date        <- the Markov state
    - running per-calendar-month sums/counts of Tmean  <- for Thornthwaite PE
    - 366 running sorted day-of-year climatology pools <- for the percentile

On the next call, it only has to:
    1. load that tiny checkpoint (state + stats, not history)
    2. run the bucket forward for the handful of NEW days you hand it
    3. update the checkpoint

The first time a village is ever requested, there's no checkpoint yet, so
it does the one full historical run (same cost as the original script) and
then never has to repeat it. Think of that exactly like the "train once"
idea from the forecast-model side of this project: pay the big cost once
per village (ideally in an offline bootstrap job, not inline in a live
request), then serve cheaply forever after.

ONE DOCUMENTED APPROXIMATION
------------------------------
The original recomputes the Thornthwaite monthly climatology from whatever
full slice of data it's handed, every run (a "whole-record" normal). This
version maintains it as running sums/counts instead, so it reflects data
seen so far rather than being retroactively recomputed over the full
timeline on every call. The numeric effect is at the rounding level (a
35-year mean barely moves when you add one more month) -- it does NOT
change the model, its parameters, or its physics. If you need strictly
bit-identical output to the original, ping me and I'll add a
"recompute-full-record" mode; it's cheap, just not needed by default.

INTERFACE
---------
    compute_village_soil_moisture(lat, lon, forecast_data, village_id=None)
        -> JSON-serializable dict

`forecast_data` is passed in directly (e.g. the `forecast` list coming out
of the forecast-correction service) -- nothing is read from a pergrid CSV.
"""

import os
import math
import time
import bisect
from datetime import date

import numpy as np
import pandas as pd

try:
    import joblib
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False
    import pickle


# ==============================================================================
# CONSTANTS — required files / directories / tunables, all up top
# ==============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# --- Required input files ---
# Resolved relative to THIS FILE's location (same fix as the forecast
# module) rather than a hardcoded absolute path, so both services agree on
# one data root regardless of where the process is launched from.
IMD_DIR = os.path.join(DATA_DIR, "IMD_parquets")
CALIB_FILE = os.path.join(DATA_DIR, "master_calibration_1D.csv")

IMD_PARQUET = {
    "pcp":  os.path.join(IMD_DIR, "imd_pcp_daily.parquet"),
    "tmax": os.path.join(IMD_DIR, "imd_tmax_daily.parquet"),
    "tmin": os.path.join(IMD_DIR, "imd_tmin_daily.parquet"),
}

# --- Where per-village checkpoints live (this replaces "recompute from 1990") ---
CHECKPOINT_DIR = os.path.join(MODELS_DIR, "soil_moisture_checkpoints")

# --- Grid / model constants (mirrors cpc_leaky_bucket.py) ---
GRID_RES = 0.25
GRID_OFFSET = 0.125
SPINUP_YEARS = 1
DOY_WINDOW = 15                 # +/- days pooled for the climatological percentile
DEFAULT_W0_FRAC = 0.5           # initial storage guess, only used on first-ever cold start
CALIB_PARAMS = ["WMAX", "Bm", "alpha_surf", "alpha_base", "melt_factor"]
HISTORY_DAYS = 30               # number of past observation days to return alongside forecast

SNOW_T_SNOW = 0.0                # <= this mean-T (degC): precip falls as snow
SNOW_T_MELT = 0.0                # >  this mean-T: melting occurs
BASE_RUNOFF_PARAMS = dict(alpha_surf=1.0, alpha_base=0.005, gamma=0.0, Bm=2.0)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ==============================================================================
# ERRORS
# ==============================================================================

class SoilMoistureError(Exception):
    error_type = "internal_error"


class InvalidLocationError(SoilMoistureError):
    error_type = "invalid_location"


class NoForcingDataError(SoilMoistureError):
    error_type = "no_forcing_data"


class CalibrationLookupError(SoilMoistureError):
    error_type = "calibration_lookup_failed"


# ==============================================================================
# SMALL UTILITIES
# ==============================================================================

def _log(verbose, msg):
    if verbose:
        print(msg)


def _validate_location(lat, lon):
    try:
        lat, lon = float(lat), float(lon)
    except (TypeError, ValueError):
        raise InvalidLocationError(f"lat/lon must be numeric, got lat={lat!r} lon={lon!r}")
    if not (-90.0 <= lat <= 90.0):
        raise InvalidLocationError(f"lat {lat} out of range [-90, 90]")
    if not (-180.0 <= lon <= 180.0):
        raise InvalidLocationError(f"lon {lon} out of range [-180, 180]")
    return lat, lon


def _dump(obj, path):
    if _JOBLIB_AVAILABLE:
        joblib.dump(obj, path)
    else:
        with open(path, "wb") as f:
            pickle.dump(obj, f)


def _load(path):
    if _JOBLIB_AVAILABLE:
        return joblib.load(path)
    with open(path, "rb") as f:
        return pickle.load(f)


# ==============================================================================
# BILINEAR GRID INTERPOLATION (unchanged logic from the original)
# ==============================================================================

def _bracket(x):
    lo = np.floor((x - GRID_OFFSET) / GRID_RES) * GRID_RES + GRID_OFFSET
    return round(lo, 3), round(lo + GRID_RES, 3)


def bilinear_weights(lat, lon):
    lat0, lat1 = _bracket(lat)
    lon0, lon1 = _bracket(lon)
    fy = min(max((lat - lat0) / GRID_RES, 0.0), 1.0)
    fx = min(max((lon - lon0) / GRID_RES, 0.0), 1.0)
    return [
        (lat0, lon0, (1 - fy) * (1 - fx)),
        (lat0, lon1, (1 - fy) * fx),
        (lat1, lon0, fy * (1 - fx)),
        (lat1, lon1, fy * fx),
    ]


_CALIB_CACHE = {"mtime": None, "df": None}


def clear_calibration_cache():
    """Clear the in-memory calibration CSV cache (cheap to rebuild; harmless to clear)."""
    _CALIB_CACHE["mtime"] = None
    _CALIB_CACHE["df"] = None


def _load_calibration():
    """Lazy-load + cache the (small) calibration CSV in memory."""
    mtime = os.path.getmtime(CALIB_FILE) if os.path.isfile(CALIB_FILE) else None
    if _CALIB_CACHE["df"] is not None and _CALIB_CACHE["mtime"] == mtime:
        return _CALIB_CACHE["df"]
    if not os.path.isfile(CALIB_FILE):
        raise CalibrationLookupError(f"calibration file not found: {CALIB_FILE}")
    calib = pd.read_csv(CALIB_FILE)
    calib = calib[calib["status"] == "ok"].copy()
    calib["lat"] = calib["lat"].round(3)
    calib["lon"] = calib["lon"].round(3)
    _CALIB_CACHE["df"] = calib
    _CALIB_CACHE["mtime"] = mtime
    return calib


def interp_calibration(lat, lon):
    calib = _load_calibration()
    lut = calib.set_index(["lat", "lon"])
    corners = bilinear_weights(lat, lon)
    acc = {p: 0.0 for p in CALIB_PARAMS}
    wsum = 0.0
    for glat, glon, w in corners:
        if w <= 0:
            continue
        try:
            row = lut.loc[(glat, glon)]
        except KeyError:
            continue
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        for p in CALIB_PARAMS:
            acc[p] += w * float(row[p])
        wsum += w
    if wsum == 0.0:
        d = (calib["lat"] - lat) ** 2 + (calib["lon"] - lon) ** 2
        if d.empty:
            raise CalibrationLookupError("calibration table is empty")
        row = calib.loc[d.idxmin()]
        return {p: float(row[p]) for p in CALIB_PARAMS}
    return {p: acc[p] / wsum for p in CALIB_PARAMS}


# ==============================================================================
# IMD HISTORICAL FORCING (bilinear) — only touched on cold start / gap-fill,
# never on a normal warm call
# ==============================================================================

def _load_imd_var(var, grid_points, start=None, end=None):
    lats = sorted({round(gp[0], 3) for gp in grid_points})
    lons = sorted({round(gp[1], 3) for gp in grid_points})
    filt = [("lat", "in", lats), ("lon", "in", lons)]
    df = pd.read_parquet(IMD_PARQUET[var], filters=filt)
    df["date"] = pd.to_datetime(df["date"])
    if start is not None:
        df = df[df["date"] >= start]
    if end is not None:
        df = df[df["date"] <= end]
    return df


def build_imd_forcing(lat, lon, start=None, end=None):
    """Bilinearly-interpolated daily IMD forcing (pcp, tmax, tmin) for [start, end]."""
    weights = bilinear_weights(lat, lon)
    grid_points = [(glat, glon) for glat, glon, w in weights if w > 0]

    series = {}
    for var in ("pcp", "tmax", "tmin"):
        long = _load_imd_var(var, grid_points, start=start, end=end)
        if long.empty:
            series[var] = pd.Series(dtype=float)
            continue
        wide = long.pivot_table(index="date", columns=["lat", "lon"], values="value")
        num = pd.Series(0.0, index=wide.index)
        den = pd.Series(0.0, index=wide.index)
        for glat, glon, w in weights:
            if w <= 0:
                continue
            col = (round(glat, 3), round(glon, 3))
            if col not in wide.columns:
                continue
            vals = wide[col]
            mask = vals.notna()
            num = num.add((vals * w).where(mask, 0.0), fill_value=0.0)
            den = den.add(pd.Series(w, index=wide.index).where(mask, 0.0), fill_value=0.0)
        series[var] = num / den.replace(0.0, np.nan)

    out = pd.DataFrame(series)
    out.index.name = "date"
    return out.sort_index()


def _earliest_imd_date():
    """Cheapest possible 'when does history start' check: min date per variable."""
    earliest = []
    for var, path in IMD_PARQUET.items():
        if not os.path.isfile(path):
            continue
        d = pd.read_parquet(path, columns=["date"])
        if not d.empty:
            earliest.append(pd.to_datetime(d["date"]).min())
    if not earliest:
        raise NoForcingDataError("no IMD parquet files found to determine history start")
    return max(earliest)  # need all 3 variables available, so take the latest "earliest"


# ==============================================================================
# PHYSICS — pure functions, no shared/mutable module-level state.
#
# NOTE: the original script set cpc.WMAX / cpc.SNOW["melt_factor"] as module
# globals per village call. That's unsafe the moment two villages' requests
# run concurrently (one village's params can leak into another's
# calculation mid-flight). Everything here takes params as arguments instead.
# ==============================================================================

def _snow_step(precip, tmean, snowpack, melt_factor):
    if tmean <= SNOW_T_SNOW:
        snowpack += precip
        rain = 0.0
    else:
        rain = precip
    melt = 0.0
    if tmean > SNOW_T_MELT and snowpack > 0.0:
        melt = min(snowpack, melt_factor * (tmean - SNOW_T_MELT))
        snowpack -= melt
    return rain + melt, snowpack


def _beta(w, wmax):
    return w / wmax


def _runoff(w, peff, wmax, p):
    ratio = max(0.0, min(1.0, w / wmax))
    r_surf = p["alpha_surf"] * (ratio ** p["Bm"]) * peff
    r_base = p["alpha_base"] * w * ratio
    return r_surf + r_base


def _bucket_day_step(w, snowpack, precip, tmean, pe, wmax, params, melt_factor, snow=True):
    """One day of the leaky-bucket recursion. Identical math to the original."""
    if snow:
        peff, snowpack = _snow_step(precip, tmean, snowpack, melt_factor)
    else:
        peff = precip

    E = _beta(w, wmax) * pe
    R = _runoff(w, peff, wmax, params)
    G = params["gamma"] * (w / wmax)
    w_new = w + peff - E - R - G

    if w_new > wmax:
        R += (w_new - wmax)
        w_new = wmax
    if w_new < 0.0:
        deficit = -w_new
        total_loss = E + R + G
        if total_loss > 0:
            E -= E / total_loss * deficit
            R -= R / total_loss * deficit
            G -= G / total_loss * deficit
        w_new = 0.0

    return w_new, snowpack, peff, E, R, G


# ==============================================================================
# THORNTHWAITE PE — running sufficient statistics instead of a full-record
# recompute every call (see the module docstring for the tradeoff this implies)
# ==============================================================================

def _new_month_stats():
    return {m: {"sum": 0.0, "count": 0} for m in range(1, 13)}


def _update_month_stats(stats, month, tmean_value):
    stats[month]["sum"] += tmean_value
    stats[month]["count"] += 1


def _thornthwaite_I_a(stats):
    i_month = []
    for m in range(1, 13):
        c = stats[m]["count"]
        norm = (stats[m]["sum"] / c) if c > 0 else 0.0
        i_month.append((max(norm, 0.0) / 5.0) ** 1.514)
    I = float(sum(i_month))
    if I <= 0:
        return 0.0, 0.0
    a = 6.75e-7 * I**3 - 7.71e-5 * I**2 + 1.792e-2 * I + 0.49239
    return I, float(a)


def _thornthwaite_pe_day(tmean, doy, lat_deg, I, a):
    if I <= 0 or tmean <= 0:
        return 0.0
    t_pos = max(tmean, 0.0)
    pe_std = 16.0 * (10.0 * t_pos / I) ** a  # mm per "standard" month

    lat = math.radians(lat_deg)
    decl = 0.4093 * math.sin(2.0 * math.pi * (284 + doy) / 365.0)
    arg = max(-1.0, min(1.0, -math.tan(lat) * math.tan(decl)))
    omega = math.acos(arg)
    N = (24.0 / math.pi) * omega
    corr = (N / 12.0) * (1.0 / 30.0)

    return max(pe_std * corr, 0.0)


# ==============================================================================
# DAY-OF-YEAR CLIMATOLOGY POOLS — built once (vectorized) on cold start,
# then updated cheaply (searchsorted + insert) per new day
# ==============================================================================

def _build_doy_pools(w_series):
    """Vectorized one-time build from a full historical w Series (cold start only)."""
    doy = w_series.index.dayofyear.values
    vals = w_series.values
    pools = {}
    for d in range(1, 367):
        lo, hi = d - DOY_WINDOW, d + DOY_WINDOW
        sel = (doy >= lo) & (doy <= hi)
        if lo < 1:
            sel |= (doy >= 366 + lo)
        if hi > 366:
            sel |= (doy <= hi - 366)
        pools[d] = np.sort(vals[sel])
    return pools


def _insert_into_doy_pools(pools, doy, w_value):
    """Cheap incremental update: insert one new value into every bin it affects."""
    for offset in range(-DOY_WINDOW, DOY_WINDOW + 1):
        d = ((doy - 1 + offset) % 366) + 1
        arr = pools[d]
        idx = np.searchsorted(arr, w_value)
        pools[d] = np.insert(arr, idx, w_value)


def _percentile_from_pool(pools, doy, w_value):
    arr = pools[doy]
    if len(arr) == 0:
        return float("nan")
    rank = np.searchsorted(arr, w_value, side="right")
    return 100.0 * rank / len(arr)


# ==============================================================================
# CHECKPOINT I/O
# ==============================================================================

def _checkpoint_path(village_id, lat, lon):
    if village_id is not None:
        tag = f"vid_{village_id}"
    else:
        tag = f"{lat:.4f}_{lon:.4f}".replace("-", "m")
    return os.path.join(CHECKPOINT_DIR, f"sm_checkpoint_{tag}.joblib")


def _load_checkpoint(path):
    if not os.path.isfile(path):
        return None
    try:
        return _load(path)
    except Exception:
        return None  # corrupt/incompatible checkpoint -> treat as cold start


def _save_checkpoint(path, checkpoint):
    _dump(checkpoint, path)


# ==============================================================================
# COLD START — the one expensive full-history run per village. Do this once
# (ideally via an offline bootstrap job across all villages), never again.
# ==============================================================================

def _cold_start(lat, lon, params, up_to_date, verbose=False):
    start = _earliest_imd_date()
    _log(verbose, f"    [cold start] building full history {start.date()} -> {up_to_date.date()}")

    forcing = build_imd_forcing(lat, lon, start=start, end=up_to_date)
    forcing = forcing.dropna(subset=["tmax", "tmin"])
    forcing["pcp"] = forcing["pcp"].fillna(0.0)
    if forcing.empty:
        raise NoForcingDataError(f"no historical IMD forcing available for ({lat}, {lon})")

    tmean = (forcing["tmax"] + forcing["tmin"]) / 2.0

    w = DEFAULT_W0_FRAC * params["WMAX"]
    snowpack = 0.0
    month_stats = _new_month_stats()
    w_track = []
    # Capture the last HISTORY_DAYS rows of output for the rolling history buffer
    recent_rows = []

    dates = forcing.index
    pcp_v = forcing["pcp"].values
    tmean_v = tmean.values

    for k in range(len(dates)):
        d = dates[k]
        month = d.month
        doy = d.dayofyear
        _update_month_stats(month_stats, month, tmean_v[k])
        I, a = _thornthwaite_I_a(month_stats)
        pe = _thornthwaite_pe_day(tmean_v[k], doy, lat, I, a)

        w, snowpack, peff, E, R, G = _bucket_day_step(
            w, snowpack, pcp_v[k], tmean_v[k], pe,
            params["WMAX"], params, params["melt_factor"], snow=True)
        w_track.append(w)

        # Build row dicts for the trailing HISTORY_DAYS window
        recent_rows.append({
            "date": d.strftime("%Y-%m-%d"),
            "P_obs": round(float(pcp_v[k]), 3),
            "Tmean": round(float(tmean_v[k]), 3),
            "PE": round(float(pe), 3),
            "P_eff": round(float(peff), 3),
            "snowpack": round(float(snowpack), 3),
            "w": round(float(w), 3),
            "E": round(float(E), 3),
            "R": round(float(R), 3),
            "G": round(float(G), 3),
            "w_frac": round(float(w) / params["WMAX"], 4),
            "sm_percentile": None,  # pools not finalized yet during cold start loop
            "is_forecast": 0,
        })
        if len(recent_rows) > HISTORY_DAYS:
            recent_rows.pop(0)

    w_series = pd.Series(w_track, index=dates)

    # Drop the spin-up period from the climatology pool (matches the original,
    # which discards `spinup_years` before computing sm_percentile), but keep
    # the *state* (w, snowpack) as-is -- that's the true physical state.
    cutoff = dates[0] + pd.DateOffset(years=SPINUP_YEARS)
    pools = _build_doy_pools(w_series[w_series.index >= cutoff])

    # Now retroactively fill in sm_percentile for the recent_rows using the
    # finalized pools and the w values we tracked.
    for row in recent_rows:
        row_date = pd.Timestamp(row["date"])
        doy = row_date.dayofyear
        pct = _percentile_from_pool(pools, doy, row["w"])
        row["sm_percentile"] = None if math.isnan(pct) else round(float(pct), 2)

    checkpoint = {
        "last_date": dates[-1],
        "w": float(w),
        "snowpack": float(snowpack),
        "month_stats": month_stats,
        "doy_pools": pools,
        "params": dict(params),
        "lat": lat, "lon": lon,
        "created_at": time.time(),
        "recent_history": recent_rows,
    }
    return checkpoint


# ==============================================================================
# INCREMENTAL STEP — used for both "fill the gap since last checkpoint" and
# "process the new forecast_data days". This is the cheap path.
# ==============================================================================

def _run_days(checkpoint, forcing_df, params, lat, tag_is_forecast=None):
    """
    Advance the bucket by exactly the days in forcing_df (must be the days
    immediately after checkpoint['last_date'], strictly increasing).
    Returns (updated_checkpoint, list_of_row_dicts).

    Observed (is_forecast == 0) rows are also appended to the checkpoint's
    recent_history rolling buffer (capped at HISTORY_DAYS entries).
    """
    w = checkpoint["w"]
    snowpack = checkpoint["snowpack"]
    month_stats = checkpoint["month_stats"]
    pools = checkpoint["doy_pools"]
    recent_history = checkpoint.get("recent_history", [])

    rows = []
    dates = forcing_df.index
    pcp_v = forcing_df["pcp"].values
    tmax_v = forcing_df["tmax"].values
    tmin_v = forcing_df["tmin"].values
    is_fc = forcing_df["is_forecast"].values if "is_forecast" in forcing_df.columns else None

    for k in range(len(dates)):
        d = dates[k]
        tmean_k = (tmax_v[k] + tmin_v[k]) / 2.0
        month, doy = d.month, d.dayofyear

        _update_month_stats(month_stats, month, tmean_k)
        I, a = _thornthwaite_I_a(month_stats)
        pe = _thornthwaite_pe_day(tmean_k, doy, lat, I, a)

        w, snowpack, peff, E, R, G = _bucket_day_step(
            w, snowpack, pcp_v[k], tmean_k, pe,
            params["WMAX"], params, params["melt_factor"], snow=True)

        _insert_into_doy_pools(pools, doy, w)
        pct = _percentile_from_pool(pools, doy, w)

        row_is_fc = int(is_fc[k]) if is_fc is not None else tag_is_forecast
        row = {
            "date": d.strftime("%Y-%m-%d"),
            "P_obs": None if pd.isna(pcp_v[k]) else round(float(pcp_v[k]), 3),
            "Tmean": round(float(tmean_k), 3),
            "PE": round(float(pe), 3),
            "P_eff": round(float(peff), 3),
            "snowpack": round(float(snowpack), 3),
            "w": round(float(w), 3),
            "E": round(float(E), 3),
            "R": round(float(R), 3),
            "G": round(float(G), 3),
            "w_frac": round(float(w) / params["WMAX"], 4),
            "sm_percentile": None if math.isnan(pct) else round(float(pct), 2),
            "is_forecast": row_is_fc,
        }
        rows.append(row)

        # Only observed (non-forecast) rows go into the persisted history
        if row_is_fc == 0 or row_is_fc is None:
            recent_history.append(row)
            if len(recent_history) > HISTORY_DAYS:
                recent_history.pop(0)

    checkpoint["last_date"] = dates[-1]
    checkpoint["w"] = float(w)
    checkpoint["snowpack"] = float(snowpack)
    checkpoint["recent_history"] = recent_history
    return checkpoint, rows


# ==============================================================================
# INPUT NORMALIZATION — accepts the forecast service's output directly
# ==============================================================================

def _normalize_forecast_rows(forecast_data):
    """
    forecast_data: list of dicts with a 'date' key and pcp/tmax/tmin values
    (accepts either raw names or '<var>_corrected' names -- corrected wins
    if both are present), optionally 'is_forecast'.
    Returns a DataFrame indexed by date, columns [pcp, tmax, tmin, is_forecast].
    """
    if not forecast_data:
        raise NoForcingDataError("forecast_data is empty")

    df = pd.DataFrame(forecast_data)
    if "date" not in df.columns:
        raise NoForcingDataError("each forecast_data row needs a 'date' key")
    df["date"] = pd.to_datetime(df["date"])

    out = pd.DataFrame(index=df["date"])
    for var in ("pcp", "tmax", "tmin"):
        corrected_col = f"{var}_corrected"
        if corrected_col in df.columns:
            out[var] = pd.to_numeric(df[corrected_col].values, errors="coerce")
        elif var in df.columns:
            out[var] = pd.to_numeric(df[var].values, errors="coerce")
        else:
            raise NoForcingDataError(f"forecast_data is missing '{var}' or '{corrected_col}'")

    out["is_forecast"] = df["is_forecast"].values if "is_forecast" in df.columns else 1
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out["pcp"] = out["pcp"].fillna(0.0)
    out[["tmax", "tmin"]] = out[["tmax", "tmin"]].interpolate(limit_direction="both")
    return out.dropna(subset=["tmax", "tmin"])


# ==============================================================================
# PUBLIC ENTRY POINT
# ==============================================================================

def compute_village_soil_moisture(lat, lon, forecast_data, *, village_id=None,
                                  verbose=False, force_cold_start=False):
    """
    Compute (or incrementally extend) daily soil moisture for one village.

    lat, lon        : village coordinates
    forecast_data   : list of dicts (the 'forecast' output of the
                      forecast-correction service works directly), each with
                      'date' and pcp/tmax/tmin (or *_corrected variants)
    village_id      : optional stable id -- if given, the checkpoint is keyed
                      by this instead of lat/lon (safer if villages can move
                      pixel-slightly between calibration runs)
    force_cold_start: ignore any existing checkpoint and rebuild from scratch
                      (use after a calibration re-fit, or to repair a
                      suspected-bad checkpoint)

    Returns a JSON-serializable dict:
      success == True:
        {
          "success": True,
          "location": {"lat":.., "lon":..},
          "cold_start": bool,          # True if this call paid the full-history cost
          "days_computed": int,
          "checkpoint_last_date": "YYYY-MM-DD",
          "soil_moisture": [ {date, w, w_frac, sm_percentile, ...}, ... ]
        }
      success == False:
        {"success": False, "error": {"type": ..., "message": ...}}
    """
    try:
        lat, lon = _validate_location(lat, lon)
        new_forcing = _normalize_forecast_rows(forecast_data)
        if new_forcing.empty:
            raise NoForcingDataError("forecast_data produced no usable rows")

        params = {**BASE_RUNOFF_PARAMS, **interp_calibration(lat, lon)}

        ckpt_path = _checkpoint_path(village_id, lat, lon)
        checkpoint = None if force_cold_start else _load_checkpoint(ckpt_path)

        cold_start_happened = False
        gap_rows = []

        if checkpoint is None:
            splice_date = new_forcing.index.min()
            checkpoint = _cold_start(lat, lon, params, splice_date - pd.Timedelta(days=1),
                                     verbose=verbose)
            cold_start_happened = True
        else:
            # Params may have changed since this checkpoint was built (e.g. a
            # recalibration run) -- if so, force a rebuild rather than silently
            # mixing old-parameter state with new-parameter dynamics.
            # Also rebuild if the checkpoint predates the recent_history feature.
            if checkpoint.get("params") != params or "recent_history" not in checkpoint:
                reason = "calibration params changed" if checkpoint.get("params") != params else "checkpoint missing recent_history"
                _log(verbose, f"    {reason} -> rebuilding checkpoint")
                splice_date = new_forcing.index.min()
                checkpoint = _cold_start(lat, lon, params, splice_date - pd.Timedelta(days=1),
                                         verbose=verbose)
                cold_start_happened = True

            gap_start = checkpoint["last_date"] + pd.Timedelta(days=1)
            gap_end = new_forcing.index.min() - pd.Timedelta(days=1)
            if gap_start <= gap_end:
                _log(verbose, f"    filling gap {gap_start.date()} -> {gap_end.date()} from IMD")
                gap_forcing = build_imd_forcing(lat, lon, start=gap_start, end=gap_end)
                gap_forcing = gap_forcing.dropna(subset=["tmax", "tmin"])
                gap_forcing["pcp"] = gap_forcing["pcp"].fillna(0.0)
                gap_forcing["is_forecast"] = 0
                if not gap_forcing.empty:
                    checkpoint, gap_rows = _run_days(checkpoint, gap_forcing, params, lat)

        # Only process forecast_data rows strictly after the checkpoint's
        # current last_date (idempotent against double-calls / retries).
        # Save the historical checkpoint to disk before we run the forecast days on it,
        # so that subsequent calls continue to begin from the end of observed historical data.
        _save_checkpoint(ckpt_path, checkpoint)

        # Retrieve the stored past-observation rows before we run forecast days
        past_history = list(checkpoint.get("recent_history", []))

        import copy
        forecast_checkpoint = copy.deepcopy(checkpoint)

        new_forcing = new_forcing[new_forcing.index > forecast_checkpoint["last_date"]]
        if new_forcing.empty and not gap_rows and not cold_start_happened:
            forecast_rows = []
        else:
            forecast_checkpoint, forecast_rows = _run_days(forecast_checkpoint, new_forcing, params, lat) \
                if not new_forcing.empty else (forecast_checkpoint, [])
            forecast_rows = gap_rows + forecast_rows

        # Combine: past observed history + current forecast/gap rows
        all_rows = past_history + forecast_rows

        return {
            "success": True,
            "location": {"lat": lat, "lon": lon},
            "cold_start": cold_start_happened,
            "history_days": len(past_history),
            "forecast_days": len(forecast_rows),
            "days_computed": len(all_rows),
            "checkpoint_last_date": forecast_checkpoint["last_date"].strftime("%Y-%m-%d"),
            "soil_moisture": all_rows,
        }

    except SoilMoistureError as e:
        return {"success": False, "error": {"type": e.error_type, "message": str(e)}}
    except Exception as e:
        return {"success": False, "error": {"type": "internal_error", "message": str(e)}}


# ==============================================================================
# Manual test entry point
# ==============================================================================

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) != 3:
        print("Usage: python village_soil_moisture_service.py <lat> <lon>")
        sys.exit(1)

    lat_arg, lon_arg = float(sys.argv[1]), float(sys.argv[2])

    # Minimal smoke-test forcing -- in real use this comes straight from
    # get_village_forecast()'s "forecast" list.
    today = pd.Timestamp(date.today())
    demo_forecast = [
        {"date": (today + pd.Timedelta(days=i)).strftime("%Y-%m-%d"),
         "pcp_corrected": 0.0, "tmax_corrected": 30.0, "tmin_corrected": 20.0,
         "is_forecast": 1}
        for i in range(16)
    ]

    out = compute_village_soil_moisture(lat_arg, lon_arg, demo_forecast, verbose=True)
    print(json.dumps(out, indent=2))