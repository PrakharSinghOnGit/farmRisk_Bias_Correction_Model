#!/usr/bin/env python3
"""
Village-level soil moisture driver.

For each pilot village:
  1. Build a continuous daily forcing series (Precip, Tmax, Tmin) by splicing:
       - <= 2021-12-31 : IMD long-term parquet, bilinearly interpolated to the
                         village lat/lon from the 4 surrounding 0.25 deg grid pts
       - 2022-01-01 -> present+16d : *_forecast_corrected columns from the
                         village's *_pergrid.csv (observed+forecast, corrected)
     The overlap is deduped by preferring the pergrid corrected values from
     2022-01-01 onward and truncating IMD at 2021-12-31.
  2. Bilinearly interpolate the calibrated leaky-bucket parameters
       (WMAX, Bm, alpha_surf, alpha_base, melt_factor) to the village lat/lon
       from master_calibration_1D.csv.
  3. Run the CPC leaky-bucket model on the full spliced forcing.
  4. Add a day-of-year climatological soil-moisture percentile column.
  5. Write one CSV per village to the output folder.

Usage:
    python run_village_soil_moisture.py
    python run_village_soil_moisture.py --village-id 1     # single village
    python run_village_soil_moisture.py --limit 3          # first N villages
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

# Import the model logic from the existing script (same folder).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import app.cpcLeakyBucket as cpc


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
ROOT          = "./"
#PERGRID_DIR   = f"{ROOT}/Bias_Correct/out"
PERGRID_DIR   = f"{ROOT}Bias_Correct/out"
POINT_DIR     = f"{ROOT}out"   # single --lat/--lon mode
IMD_DIR       = f"{ROOT}data/IMD_parquets"
#VILLAGES_FILE = f"{ROOT}/Pilot_80_Villages"
VILLAGES_FILE = f"{ROOT}Pilot_80_Villages"
CALIB_FILE    = f"{ROOT}data/master_calibration_1D.csv"
OUT_DIR       = f"{ROOT}out"

IMD_PARQUET = {
    "pcp":  f"{IMD_DIR}/imd_pcp_daily.parquet",
    "tmax": f"{IMD_DIR}/imd_tmax_daily.parquet",
    "tmin": f"{IMD_DIR}/imd_tmin_daily.parquet",
}

# 'lon lat elev' grid file for FAO-56 Penman-Monteith (pressure / gamma).
ELEV_FILE = f"{ROOT}data/grid_elevation.parquet"

# District-level crop sowing/harvest calendar (for --plant-day default).
CROP_CALENDAR_FILE = f"{ROOT}data/crop_calendar_parsed.csv"

# Map model crop keys (cpc.CROP_TABLE) to keyword(s) matched (case-
# insensitive substring) against the calendar's messy 'Crop' names.
CROP_CALENDAR_KEYWORDS = {
    "wheat": ["wheat"], "wheat_spring": ["wheat"],
    "rice": ["rice", "paddy"],
    "maize": ["maize"], "maize_sweet": ["maize"],
    "sorghum": ["sorghum", "jowar", "juvar"],
    "millet": ["millet", "bajra", "bajri", "bajara", "ragi"],
    "cotton": ["cotton"],
    "sugarcane": ["sugarcane"],
    "groundnut": ["groundnut", "ground nut", "g'nut"],
    "soybean": ["soybean", "soyabean"],
    "chickpea": ["chickpea", "gram", "chana"],
    "mustard": ["mustard", "toria", "rapeseed", "rape seed"],
    "potato": ["potato"], "tomato": ["tomato"], "grass": ["fodder", "grass"],
}

GRID_RES   = 0.25
CORRECTED_LOOKBACK = 5   # use forecast-corrected from (present - N) days onward;
                        # IMD parquet fills everything strictly before that.
                        # "present" = last observed (is_forecast==0) date in the file.
CALIB_PARAMS = ["WMAX", "Bm", "alpha_surf", "alpha_base", "melt_factor"]
RETURN_TAIL_ROWS = 50


# ----------------------------------------------------------------------
# Default planting DOY from the district crop calendar
# ----------------------------------------------------------------------
def _window_mid_doy(rows):
    """Midpoint DOY of the sowing window across matching calendar rows.

    Uses Sow_From_(Day,Mon) .. Sow_To_(Day,Mon); rows missing the 'to'
    date fall back to the 'from' date. Handles windows that wrap the year
    end (e.g. Dec -> Jan). Returns an int DOY in 1..365 or None.
    """
    import datetime as _dt
    mids = []
    ref_year = 2001  # non-leap reference for DOY
    for _, r in rows.iterrows():
        fd, fm = r["Sow_From_Day"], r["Sow_From_Mon"]
        td, tm = r["Sow_To_Day"], r["Sow_To_Mon"]
        if pd.isna(fd) or pd.isna(fm):
            continue
        try:
            d0 = _dt.date(ref_year, int(fm), int(fd))
        except ValueError:
            continue
        if pd.isna(td) or pd.isna(tm):
            d1 = d0                       # single date -> use it directly
        else:
            try:
                d1 = _dt.date(ref_year, int(tm), int(td))
            except ValueError:
                d1 = d0
        doy0 = d0.timetuple().tm_yday
        doy1 = d1.timetuple().tm_yday
        if doy1 < doy0:                   # window wraps year-end
            doy1 += 365
        mid = int(round((doy0 + doy1) / 2.0))
        if mid > 365:
            mid -= 365
        mids.append(mid)
    if not mids:
        return None
    # Average the per-row midpoints (circular mean would be ideal but the
    # windows here are tight; arithmetic mean on unwrapped DOYs is fine).
    return int(round(np.mean(mids)))


def default_plant_day(crop, district=None, calendar_path=CROP_CALENDAR_FILE):
    """
    Approximate planting DOY for `crop` from the district crop calendar.
    Priority: district -> state (of that district) -> national, using the
    sowing-window midpoint. Returns (doy, scope_str) or (None, reason).
    """
    if not os.path.exists(calendar_path):
        return None, f"calendar not found: {calendar_path}"
    cal = pd.read_csv(calendar_path)
    keywords = CROP_CALENDAR_KEYWORDS.get(crop, [crop])
    cl = cal["Crop"].fillna("").str.lower()
    crop_mask = pd.Series(False, index=cal.index)
    for kw in keywords:
        crop_mask |= cl.str.contains(kw, regex=False)
    sub = cal[crop_mask]
    if sub.empty:
        return None, f"no '{crop}' rows in calendar"

    # District scope (and resolve its state for the state fallback).
    state = None
    if district is not None:
        dmask = sub["District"].fillna("").str.lower() == district.lower()
        drows = sub[dmask]
        if not drows.empty:
            doy = _window_mid_doy(drows)
            if doy is not None:
                return doy, f"district={district}"
        # resolve state from the full calendar even if crop rows absent
        dall = cal[cal["District"].fillna("").str.lower() == district.lower()]
        if not dall.empty:
            state = dall["State"].iloc[0]

    # State scope.
    if state is not None:
        smask = sub["State"].fillna("").str.lower() == str(state).lower()
        srows = sub[smask]
        if not srows.empty:
            doy = _window_mid_doy(srows)
            if doy is not None:
                return doy, f"state={state}"

    # National scope.
    doy = _window_mid_doy(sub)
    if doy is not None:
        return doy, "national"
    return None, f"no usable sowing dates for '{crop}'"


# ----------------------------------------------------------------------
# Bilinear interpolation on a regular 0.25 deg grid
# ----------------------------------------------------------------------
def _bracket(x):
    """Return (lo, hi) 0.25 deg grid centers bracketing coordinate x."""
    lo = np.floor((x - 0.125) / GRID_RES) * GRID_RES + 0.125
    return round(lo, 3), round(lo + GRID_RES, 3)


def bilinear_weights(lat, lon):
    """
    Return list of (glat, glon, weight) for the 4 surrounding grid points.
    Standard bilinear weighting on the unit cell.
    """
    lat0, lat1 = _bracket(lat)
    lon0, lon1 = _bracket(lon)
    fy = (lat - lat0) / GRID_RES
    fx = (lon - lon0) / GRID_RES
    fy = min(max(fy, 0.0), 1.0)
    fx = min(max(fx, 0.0), 1.0)
    return [
        (lat0, lon0, (1 - fy) * (1 - fx)),
        (lat0, lon1, (1 - fy) * fx),
        (lat1, lon0, fy * (1 - fx)),
        (lat1, lon1, fy * fx),
    ]


def interp_calibration(calib, lat, lon):
    """
    Bilinearly interpolate calibrated params to (lat, lon).
    Missing corners are dropped and remaining weights renormalized; if no
    corner is available, fall back to the nearest valid grid cell.
    """
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
        if isinstance(row, pd.DataFrame):      # duplicate grid -> take first
            row = row.iloc[0]
        for p in CALIB_PARAMS:
            acc[p] += w * float(row[p])
        wsum += w
    if wsum == 0.0:
        # nearest valid cell fallback
        d = (calib["lat"] - lat) ** 2 + (calib["lon"] - lon) ** 2
        row = calib.loc[d.idxmin()]
        return {p: float(row[p]) for p in CALIB_PARAMS}
    return {p: acc[p] / wsum for p in CALIB_PARAMS}


# ----------------------------------------------------------------------
# IMD historical forcing (bilinear over the 4 surrounding grid points)
# ----------------------------------------------------------------------
def load_imd_var(var, grid_points, end):
    """
    Read one IMD parquet, keep only the needed grid points and dates <= end.
    Returns a long DataFrame [lat, lon, date, value].
    Uses a pushdown filter on the grid points for memory efficiency.
    """
    lats = sorted({round(gp[0], 3) for gp in grid_points})
    lons = sorted({round(gp[1], 3) for gp in grid_points})
    filt = [("lat", "in", lats), ("lon", "in", lons)]
    df = pd.read_parquet(IMD_PARQUET[var], filters=filt)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= end]
    return df


def build_imd_forcing(lat, lon, end):
    """
    Build bilinearly-interpolated daily IMD forcing (pcp, tmax, tmin) for a
    village, for all dates <= `end`. Returns DataFrame indexed by date
    with columns pcp, tmax, tmin.
    """
    weights = bilinear_weights(lat, lon)
    grid_points = [(glat, glon) for glat, glon, w in weights if w > 0]

    series = {}
    for var in ("pcp", "tmax", "tmin"):
        long = load_imd_var(var, grid_points, end)
        # pivot to date x (lat,lon)
        wide = long.pivot_table(index="date", columns=["lat", "lon"],
                                values="value")
        # weighted sum over available corners, renormalizing weights per-date
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
            den = den.add(pd.Series(w, index=wide.index).where(mask, 0.0),
                          fill_value=0.0)
        series[var] = (num / den.replace(0.0, np.nan))
    out = pd.DataFrame(series)
    out.index.name = "date"
    return out


# ----------------------------------------------------------------------
# Pergrid (2022 -> present+16d) corrected forcing
# ----------------------------------------------------------------------
def build_pergrid_forcing(pergrid_path):
    """Return DataFrame indexed by date: pcp, tmax, tmin, is_forecast."""
    df = pd.read_csv(pergrid_path, parse_dates=["date"])
    df = df.rename(columns={
        "pcp_forecast_corrected":  "pcp",
        "tmax_forecast_corrected": "tmax",
        "tmin_forecast_corrected": "tmin",
    })
    df = df.set_index("date").sort_index()
    keep = df[["pcp", "tmax", "tmin", "is_forecast"]].copy()
    return keep


# ----------------------------------------------------------------------
# Splice IMD + pergrid into a single continuous forcing record
# ----------------------------------------------------------------------
def build_forcing(lat, lon, pergrid_path):
    per = build_pergrid_forcing(pergrid_path)

    # "present" = last observed (is_forecast==0) date; corrected window is
    # (present - CORRECTED_LOOKBACK) .. end-of-file (present + 16d).
    present = per.index[per["is_forecast"] == 0].max()
    splice_cut = present - pd.Timedelta(days=CORRECTED_LOOKBACK)

    imd = build_imd_forcing(lat, lon, end=splice_cut - pd.Timedelta(days=1))
    imd = imd[imd.index < splice_cut]
    imd["is_forecast"] = 0

    per = per[per.index >= splice_cut]

    forcing = pd.concat([imd, per], axis=0)
    forcing = forcing[~forcing.index.duplicated(keep="last")].sort_index()
    # fill any gaps in the daily index
    full_idx = pd.date_range(forcing.index.min(), forcing.index.max(), freq="D")
    forcing = forcing.reindex(full_idx)
    forcing.index.name = "date"
    forcing["pcp"] = forcing["pcp"].fillna(0.0)
    forcing[["tmax", "tmin"]] = forcing[["tmax", "tmin"]].interpolate(
        limit_direction="both")
    forcing["is_forecast"] = forcing["is_forecast"].fillna(0).astype(int)
    return forcing


# ----------------------------------------------------------------------
# Day-of-year climatological percentile of storage w
# ----------------------------------------------------------------------
def doy_percentile(w, window=15):
    """
    For each day, percentile of w within a +/- `window`-day day-of-year
    climatology pooled across all years (CPC-style). Returns 0-100 Series.
    """
    s = w.dropna()
    doy = s.index.dayofyear.values
    vals = s.values
    n = len(s)
    pct = np.full(n, np.nan)

    # Pre-bin values by day-of-year for speed.
    by_doy = {}
    for d in range(1, 367):
        lo, hi = d - window, d + window
        sel = ((doy >= lo) & (doy <= hi))
        # wrap-around at year boundary
        if lo < 1:
            sel |= (doy >= 366 + lo)
        if hi > 366:
            sel |= (doy <= hi - 366)
        by_doy[d] = np.sort(vals[sel])

    for i in range(n):
        ref = by_doy[doy[i]]
        # percentile = fraction of climatology <= this value
        rank = np.searchsorted(ref, vals[i], side="right")
        pct[i] = 100.0 * rank / len(ref)

    return pd.Series(pct, index=s.index)


# ----------------------------------------------------------------------
# Per-village processing
# ----------------------------------------------------------------------
def run_point(pergrid_path, taluka, village, out_tag, lat, lon, calib,
              elev_table=None, crop=None, plant_day=1, daysbefore=None, 
              out_dir=OUT_DIR, save_csv=False):
    """Core pipeline for one point; out_tag names the output file sm_<tag>.csv."""
    forcing = build_forcing(lat, lon, pergrid_path)
    params_i = interp_calibration(calib, lat, lon)

    params = dict(cpc.PARAMS)                 # defaults (incl. gamma=0)
    params.update({k: params_i[k] for k in ("Bm", "alpha_surf", "alpha_base")})
    # WMAX and melt_factor are passed functionally to run_model.
    wmax = float(params_i["WMAX"])
    snow_params = dict(cpc.SNOW)
    snow_params["melt_factor"] = float(params_i["melt_factor"])

    # Elevation for FAO-56 Penman-Monteith (nearest grid cell, 0 if no file).
    if elev_table is not None:
        table, elat, elon, eelev = elev_table
        elev = cpc.lookup_elevation(lat, lon, table, elat, elon, eelev)
    else:
        elev = 0.0

    tmean = (forcing["tmax"] + forcing["tmin"]) / 2.0
    pe = cpc.penman_monteith_eto(forcing["tmax"], forcing["tmin"],
                                 forcing.index, lat, elev)

    # FAO-56 crop coefficient: ETc = Kc * ETo (only if a crop is requested).
    if crop is not None:
        kc = cpc.build_kc_series(
            forcing.index, forcing["tmax"], forcing["tmin"],
            crop=crop, plant_day=plant_day, u2=cpc.PM["u2"],
        )
        pe = pe * kc

    res = cpc.run_model(
        forcing["pcp"], tmean, pe, lat,
        params=params, snow=True, w0_frac=0.5, spinup_years=1,
        irrig_daysbefore=daysbefore,
        wmax=wmax,
        snow_params=snow_params,
    )

    # attach is_forecast (model drops 1 spin-up year)
    res = res.join(forcing[["is_forecast"]], how="left")
    res["sm_percentile"] = doy_percentile(res["w"])

    res.insert(0, "village", village)
    res.insert(1, "taluka", taluka)
    res.insert(2, "lat", lat)
    res.insert(3, "lon", lon)

    if save_csv and out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"sm_{out_tag}.csv")
        res.to_csv(out_path, float_format="%.3f", index_label="date")

    # Filter tail rows and return as JSON records
    tail_df = res.tail(RETURN_TAIL_ROWS).reset_index()
    if 'date' in tail_df.columns:
        if pd.api.types.is_datetime64_any_dtype(tail_df['date']):
            tail_df['date'] = tail_df['date'].dt.strftime('%Y-%m-%d')
        else:
            tail_df['date'] = pd.to_datetime(tail_df['date']).dt.strftime('%Y-%m-%d')
            
    return tail_df.to_json(orient='records')


def process_village(vid, taluka, village, lat, lon, calib,
                    elev_table=None, crop=None, plant_day=1, daysbefore=None,
                    pergrid_dir=None, out_dir=OUT_DIR, save_csv=False):
    pergrid_path = find_pergrid(vid, village, pergrid_dir=pergrid_dir)
    if pergrid_path is None:
        from app.forcast import run_forecast_pipeline
        run_forecast_pipeline(lat=lat, lon=lon, output_dir=pergrid_dir)
        pergrid_path = os.path.join(pergrid_dir, f"forecast_{lat:.4f}_{lon:.4f}.csv")
        
    return run_point(pergrid_path, taluka, village, f"{vid}_{village}",
                     lat, lon, calib, elev_table=elev_table,
                     crop=crop, plant_day=plant_day, daysbefore=daysbefore,
                     out_dir=out_dir, save_csv=save_csv)


def find_pergrid(vid, village, pergrid_dir=None):
    """Locate the village pergrid csv, tolerant of name casing."""
    if pergrid_dir is None:
        pergrid_dir = PERGRID_DIR
    cands = [
        f"{vid}_{village.lower()}_pergrid.csv",
        f"{vid}_{village}_pergrid.csv",
    ]
    for c in cands:
        p = os.path.join(pergrid_dir, c)
        if os.path.exists(p):
            return p
    # fallback: glob by id prefix
    import glob
    hits = glob.glob(os.path.join(pergrid_dir, f"{vid}_*_pergrid.csv"))
    return hits[0] if hits else None


# ----------------------------------------------------------------------
def run_soil_moisture_pipeline(
    village_id=None,
    limit=None,
    lat=None,
    lon=None,
    crop=None,
    plant_day=None,
    district=None,
    daysbefore=None,
    calib_file=CALIB_FILE,
    elev_file=ELEV_FILE,
    crop_calendar_file=CROP_CALENDAR_FILE,
    point_dir=POINT_DIR,
    out_dir=OUT_DIR,
    pergrid_dir=None,
    villages_file=None,
    save_csv=False,
):
    import time
    t0 = time.time()

    if pergrid_dir is None:
        pergrid_dir = globals().get("PERGRID_DIR", f"{ROOT}Bias_Correct/out")
    if villages_file is None:
        villages_file = globals().get("VILLAGES_FILE", f"{ROOT}Pilot_80_Villages")

    # Resolve planting DOY when a crop is requested but plant_day omitted.
    if crop is not None and plant_day is None:
        day = default_plant_day(crop, district=district, calendar_path=crop_calendar_file)
        if day is not None:
            plant_day = day
        else:
            plant_day = 1
    elif plant_day is None:
        plant_day = 1   # no crop -> value is unused anyway

    calib = pd.read_csv(calib_file)
    calib = calib[calib["status"] == "ok"].copy()
    calib["lat"] = calib["lat"].round(3)
    calib["lon"] = calib["lon"].round(3)

    # Load the elevation grid once (shared across all points).
    elev_table = None
    if os.path.exists(elev_file):
        elev_table = cpc.load_elevation_file(elev_file)

    # --- single-point mode: --lat/--lon, forecast_<lat4>_<lon4>.csv ---
    if lat is not None or lon is not None:
        if lat is None or lon is None:
            sys.exit("ERROR: --lat and --lon must be given together")
        pergrid_path = os.path.join(
            point_dir, f"forecast_{lat:.4f}_{lon:.4f}.csv")
        if not os.path.exists(pergrid_path):
            from app.forcast import run_forecast_pipeline
            print(f"Forecast file not found. Generating forecast for lat={lat:.4f}, lon={lon:.4f}...")
            run_forecast_pipeline(lat=lat, lon=lon, output_dir=point_dir)
        tag = f"{lat:.4f}_{lon:.4f}"
        
        duration = time.time() - t0
        try:
            from app.stats import update_stats
            update_stats("soil_moisture", duration)
        except Exception as e:
            print(f"[STATS WARNING] Error logging stats: {e}")
            
        return run_point(pergrid_path, "point", "point", tag,
                         lat, lon, calib, elev_table=elev_table,
                         crop=crop, plant_day=plant_day,
                         daysbefore=daysbefore, out_dir=out_dir, save_csv=save_csv)

    villages = pd.read_csv(
        villages_file, sep=r"\s+", header=None,
        names=["id", "taluka", "village", "lat", "lon"], engine="python",
    )

    if village_id is not None:
        villages = villages[villages["id"] == village_id]
    if limit:
        villages = villages.head(limit)

    results_dict = {}
    for _, r in villages.iterrows():
        res_json = process_village(int(r.id), r.taluka, r.village,
                                   float(r.lat), float(r.lon), calib,
                                   elev_table=elev_table, crop=crop,
                                   plant_day=plant_day,
                                   daysbefore=daysbefore,
                                   pergrid_dir=pergrid_dir,
                                   out_dir=out_dir,
                                   save_csv=save_csv)
        if res_json is not None:
            import json
            results_dict[f"{r.id}_{r.village}"] = json.loads(res_json)
            
    import json
    res = json.dumps(results_dict)
    duration = time.time() - t0
    try:
        from app.stats import update_stats
        update_stats("soil_moisture", duration)
    except Exception as e:
        print(f"[STATS WARNING] Error logging stats: {e}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--village-id", type=int, help="run a single village id")
    ap.add_argument("--limit", type=int, help="run only first N villages")
    ap.add_argument("--lat", type=float, help="run a single point by latitude")
    ap.add_argument("--lon", type=float, help="run a single point by longitude")
    # FAO-56 crop coefficient + irrigation (forwarded to cpc_leaky_bucket)
    ap.add_argument("--crop", choices=sorted(cpc.CROP_TABLE.keys()),
                    help="apply FAO-56 seasonal Kc for this crop")
    ap.add_argument("--plant-day", type=int, default=None,
                    help="growing-season start day-of-year (with --crop). "
                         "If omitted, derived from the district crop calendar")
    ap.add_argument("--district",
                    help="district name for crop-calendar plant-day lookup")
    ap.add_argument("--daysbefore", type=int,
                    help="deficit-refill irrigation this many days before "
                         "end of record")
    args = ap.parse_args()

    res_json = run_soil_moisture_pipeline(
        village_id=args.village_id,
        limit=args.limit,
        lat=args.lat,
        lon=args.lon,
        crop=args.crop,
        plant_day=args.plant_day,
        district=args.district,
        daysbefore=args.daysbefore,
        save_csv=True,
    )


if __name__ == "__main__":
    main()
