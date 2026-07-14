#!/usr/bin/env python3
"""
CPC one-layer "leaky bucket" soil moisture model (Huang et al. 1996;
van den Dool et al. 2003; Fan & van den Dool 2004) with a simple
temperature-threshold snow module (after Fan 2019).

Forcing: daily Precip, Tmax, Tmin point files in the IMD format
    columns: [year month day value]   (whitespace separated)
    Precip in mm, Tmax/Tmin in degC.

Water balance (daily, single 1.6 m layer, wmax = 760 mm):
    w_{t+1} = w_t + Peff - E - R - G
where
    Peff = rain + snowmelt
    E    = beta(w) * PE        (PE: FAO-56 Penman-Monteith ETo, reduced-set)
    R    = surface + base runoff (CPC Bm-style, function of w/wmax)
    G    = linear groundwater loss = gamma * (w/wmax)

Output: CSV with daily date, forcing, snowpack, w, E, R, G, and w as a
fraction of wmax.

Usage:
    python cpc_leaky_bucket.py \
        --pcp  /media/urmin/data1/.../RT_pcp/data_36.875_74.625 \
        --tmax /media/urmin/data1/.../RT_tmax/data_36.875_74.625 \
        --tmin /media/urmin/data1/.../RT_tmin/data_36.875_74.625 \
        --lat  36.875 \
        --out  sm_36.875_74.625.csv

If --pcp etc. are not given, the script derives the three paths from a
single --base point file path by swapping the RT_pcp/RT_tmax/RT_tmin
folder names, and reads --lat from the filename if not supplied.
"""

import argparse
import os
import re
import sys
import math
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Model parameters (CPC operational defaults; see Huang et al. 1996,
# Table-style tuned values, van den Dool et al. 2003). Tunable.
# ----------------------------------------------------------------------
WMAX = 760.0        # max water holding capacity (mm)  -> 1.6 m, porosity 0.47

# Runoff parameters (surface + base). CPC parameterizes runoff as a
# nonlinear function of the storage ratio. Bm controls surface runoff
# curvature; alpha_base the linear baseflow.
PARAMS = dict(
    Bm=2.0,          # surface-runoff exponent (storage-ratio sensitivity)
    alpha_surf=1.0,  # surface-runoff scaling on instantaneous input
    alpha_base=0.005,# base-runoff (slow drainage) linear coefficient
    gamma=0.0,       # groundwater loss coefficient (set 0 -> folded into base)
    # Thornthwaite uses long-term monthly normals for the heat index; we
    # build the heat index from the record itself.
)

# Snow module (temperature-threshold, Fan 2019 style)
SNOW = dict(
    t_snow=0.0,      # below this mean-T (degC) precip falls as snow
    t_melt=0.0,      # above this mean-T melting occurs
    melt_factor=2.5, # degree-day melt factor (mm / degC / day)
)


# ----------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------
def read_point_file(path):
    """Read a [year month day value] whitespace file -> indexed Series."""
    if not os.path.exists(path):
        sys.exit(f"ERROR: file not found: {path}")
    df = pd.read_csv(
        path, sep=r"\s+", header=None,
        names=["year", "month", "day", "value"],
        engine="python",
    )
    # Build a proper datetime index, dropping any malformed dates.
    dt = pd.to_datetime(
        dict(year=df.year, month=df.month, day=df.day),
        errors="coerce",
    )
    s = pd.Series(df.value.values, index=dt, name=os.path.basename(path))
    s = s[~s.index.isna()]
    # Some IMD point files contain repeated dates; keep the last value per
    # date so every downstream consumer gets a unique DatetimeIndex
    # (duplicate labels break pandas reindex/align operations).
    s = s[~s.index.duplicated(keep="last")]
    return s


def derive_paths(base):
    """Given one RT_* point file path, derive the other two by folder swap."""
    paths = {}
    for var, folder in (("pcp", "RT_pcp"), ("tmax", "RT_tmax"),
                        ("tmin", "RT_tmin")):
        if "RT_pcp" in base:
            paths[var] = base.replace("RT_pcp", folder)
        elif "RT_tmax" in base:
            paths[var] = base.replace("RT_tmax", folder)
        elif "RT_tmin" in base:
            paths[var] = base.replace("RT_tmin", folder)
        else:
            sys.exit("ERROR: --base must contain RT_pcp / RT_tmax / RT_tmin")
    return paths["pcp"], paths["tmax"], paths["tmin"]


def lat_from_filename(path):
    """data_<lat>_<lon> -> lat as float, or None."""
    m = re.search(r"data_(-?\d+\.?\d*)_(-?\d+\.?\d*)", os.path.basename(path))
    return float(m.group(1)) if m else None


def latlon_from_filename(path):
    """data_<lat>_<lon> -> (lat, lon) as floats, or (None, None)."""
    m = re.search(r"data_(-?\d+\.?\d*)_(-?\d+\.?\d*)", os.path.basename(path))
    if not m:
        return None, None
    return float(m.group(1)), float(m.group(2))


def load_elevation_file(path):
    """
    Read a whitespace 'lon lat elev' grid file OR a parquet file
    -> dict keyed by rounded (lat, lon) for fast nearest-cell lookup.
    Returns (dict, grid_res).
    """
    if not os.path.exists(path):
        sys.exit(f"ERROR: elevation file not found: {path}")
    if path.endswith(".parquet"):
        df = pd.read_parquet(path, memory_map=False)
        lats = df["lat"].to_numpy()
        lons = df["lon"].to_numpy()
        elevs = df["elevation"].to_numpy()
    else:
        arr = np.loadtxt(path)
        if arr.ndim != 2 or arr.shape[1] < 3:
            sys.exit("ERROR: elevation file must have columns: lon lat elev")
        lons, lats, elevs = arr[:, 0], arr[:, 1], arr[:, 2]
    table = {(round(la, 3), round(lo, 3)): el
             for lo, la, el in zip(lons, lats, elevs)}
    return table, lats, lons, elevs


def lookup_elevation(lat, lon, table, lats, lons, elevs):
    """
    Look up elevation for (lat, lon). Exact match on 3-dp rounding first;
    otherwise nearest grid cell by Euclidean distance in lat/lon.
    """
    key = (round(lat, 3), round(lon, 3))
    if key in table:
        return table[key]
    d2 = (lats - lat) ** 2 + (lons - lon) ** 2
    return float(elevs[int(np.argmin(d2))])


# ----------------------------------------------------------------------
# Reference evapotranspiration: FAO-56 Penman-Monteith (Allen et al. 1998)
#
# Full FAO-56 grass-reference ETo (Eq. 6) using a temperature-only
# ("reduced-set") input stream, exactly as FAO-56 prescribes when the
# station reports only Tmax/Tmin (Chapter 3 / Annex 6). This is the same
# strategy VIC's MTCLIM preprocessor uses: solar radiation is derived from
# the diurnal temperature range (Hargreaves radiation, Eq. 50), actual
# vapour pressure from Tmin as a dewpoint proxy (Eq. 48), net radiation
# from the estimated Rs and ea, wind defaulted to 2 m/s, and G = 0 daily.
#
# Equation numbers below refer to FAO Irrigation & Drainage Paper 56.
# ----------------------------------------------------------------------
PM = dict(
    kRs=0.16,        # Hargreaves radiation coefficient (0.16 interior, 0.19 coastal)
    u2=2.0,          # default wind speed at 2 m (m/s) when unavailable
    albedo=0.23,     # grass reference albedo
    G=0.0,           # soil heat flux, ~0 at daily step
)
GSC = 0.0820         # solar constant (MJ m-2 min-1)
SIGMA = 4.903e-9     # Stefan-Boltzmann (MJ K-4 m-2 day-1)


def sat_vapour_pressure(t):
    """Saturation vapour pressure e0(T) (kPa), FAO-56 Eq. 11."""
    return 0.6108 * np.exp(17.27 * t / (t + 237.3))


def penman_monteith_eto(tmax, tmin, dates, lat_deg, elev_m, pm=PM):
    """
    Daily FAO-56 grass-reference ETo (mm/day) from Tmax, Tmin, latitude
    and elevation only. Returns a Series aligned to `dates`.
    """
    tmx = np.asarray(tmax, dtype=float)
    tmn = np.asarray(tmin, dtype=float)
    tmean = (tmx + tmn) / 2.0

    # --- Atmospheric pressure (Eq. 7) and psychrometric constant (Eq. 8) ---
    P = 101.3 * ((293.0 - 0.0065 * elev_m) / 293.0) ** 5.26      # kPa
    gamma = 0.000665 * P                                          # kPa/degC

    # --- Vapour pressure terms ---
    es = (sat_vapour_pressure(tmx) + sat_vapour_pressure(tmn)) / 2.0  # Eq. 12
    ea = sat_vapour_pressure(tmn)                                     # Eq. 48 (Tdew~Tmin)
    # Slope of saturation vapour pressure curve (Eq. 13)
    delta = (4098.0 * sat_vapour_pressure(tmean)
             / (tmean + 237.3) ** 2)

    # --- Extraterrestrial radiation Ra (Eq. 21-25) ---
    lat = math.radians(lat_deg)
    doy = np.asarray(dates.dayofyear, dtype=float)
    dr = 1.0 + 0.033 * np.cos(2.0 * math.pi / 365.0 * doy)        # Eq. 23
    decl = 0.409 * np.sin(2.0 * math.pi / 365.0 * doy - 1.39)     # Eq. 24
    ws_arg = np.clip(-np.tan(lat) * np.tan(decl), -1.0, 1.0)
    ws = np.arccos(ws_arg)                                        # sunset hour angle (Eq. 25)
    Ra = (24.0 * 60.0 / math.pi) * GSC * dr * (
        ws * math.sin(lat) * np.sin(decl)
        + math.cos(lat) * np.cos(decl) * np.sin(ws))             # Eq. 21, MJ m-2 day-1

    # --- Solar radiation from temperature range (Hargreaves, Eq. 50) ---
    dt_range = np.clip(tmx - tmn, 0.0, None)
    Rs = pm["kRs"] * np.sqrt(dt_range) * Ra
    # Clear-sky radiation (Eq. 37) for the longwave cloudiness term
    Rso = (0.75 + 2e-5 * elev_m) * Ra
    Rs = np.minimum(Rs, Rso)                                      # Rs <= Rso

    # --- Net shortwave (Eq. 38) and net longwave (Eq. 39) ---
    Rns = (1.0 - pm["albedo"]) * Rs
    with np.errstate(divide="ignore", invalid="ignore"):
        rs_rso = np.where(Rso > 0, np.clip(Rs / Rso, 0.0, 1.0), 0.0)
    tmaxK = tmx + 273.16
    tminK = tmn + 273.16
    Rnl = (SIGMA * (tmaxK ** 4 + tminK ** 4) / 2.0
           * (0.34 - 0.14 * np.sqrt(np.clip(ea, 0.0, None)))
           * (1.35 * rs_rso - 0.35))                             # MJ m-2 day-1
    Rn = Rns - Rnl                                               # Eq. 40

    # --- FAO-56 Penman-Monteith reference ETo (Eq. 6) ---
    u2 = pm["u2"]
    num = (0.408 * delta * (Rn - pm["G"])
           + gamma * (900.0 / (tmean + 273.0)) * u2 * (es - ea))
    den = delta + gamma * (1.0 + 0.34 * u2)
    eto = num / den
    eto = np.maximum(eto, 0.0)
    return pd.Series(eto, index=dates)


# ----------------------------------------------------------------------
# Crop coefficient: FAO-56 single crop coefficient, Kc (Chapter 6)
#
# ETc = Kc * ETo. Kc follows the FAO-56 seasonal curve (Fig. 34, Eq. 66):
#   - initial stage   : Kc = Kc_ini            (constant)
#   - development     : linear ramp Kc_ini -> Kc_mid
#   - mid-season      : Kc = Kc_mid            (constant)
#   - late season     : linear ramp Kc_mid -> Kc_end
#   - off-season      : Kc = Kc_off (bare soil / fallow)
#
# Kc_mid and Kc_end are climate-adjusted (Eq. 62 / 65) from the tabulated
# values for RHmin != 45% or u2 != 2 m/s. RHmin is estimated from Tmin as a
# dewpoint proxy: RHmin = 100 * e0(Tmin)/e0(Tmax)  (consistent with the PM
# ea = e0(Tmin) assumption used for ETo above).
#
# Table 11 (stage lengths, days) and Table 12 (Kc_ini/mid/end + crop
# height h, m) values below are FAO-56 defaults; Indian-relevant crops.
# ----------------------------------------------------------------------
# crop -> (L_ini, L_dev, L_mid, L_late, Kc_ini, Kc_mid, Kc_end, h_m)
CROP_TABLE = {
    "wheat":     (30,  140, 40,  30,  0.30, 1.15, 0.30, 1.0),   # winter wheat
    "wheat_spring": (20, 25, 60, 30, 0.30, 1.15, 0.30, 1.0),
    "rice":      (30,  30,  60,  30,  1.05, 1.20, 0.75, 1.0),   # paddy
    "maize":     (30,  40,  50,  30,  0.30, 1.20, 0.60, 2.0),   # grain maize
    "maize_sweet": (20, 25, 25, 10, 0.30, 1.15, 1.05, 1.5),
    "sorghum":   (20,  35,  40,  30,  0.30, 1.10, 0.55, 1.5),
    "millet":    (15,  25,  40,  25,  0.30, 1.00, 0.30, 1.5),
    "cotton":    (30,  50,  60,  55,  0.35, 1.18, 0.60, 1.35),
    "sugarcane": (35,  60,  190, 120, 0.40, 1.25, 0.75, 3.0),
    "groundnut": (25,  35,  45,  25,  0.40, 1.15, 0.60, 0.4),
    "soybean":   (20,  30,  60,  25,  0.40, 1.15, 0.50, 0.5),
    "chickpea":  (20,  30,  35,  25,  0.40, 1.00, 0.35, 0.4),
    "mustard":   (20,  35,  45,  25,  0.35, 1.10, 0.35, 0.6),   # rapeseed/canola
    "potato":    (25,  30,  45,  30,  0.50, 1.15, 0.75, 0.6),
    "tomato":    (30,  40,  40,  25,  0.60, 1.15, 0.80, 0.6),
    "grass":     (10,  20,  60,  30,  0.30, 1.00, 0.85, 0.5),   # generic pasture
}
KC_OFF = 0.20   # off-season / bare-soil-fallow Kc


def _clip01(x):
    return max(0.0, min(1.0, x))


def climate_adjust_kc(kc_tab, h_m, rhmin_mean, u2, stage):
    """
    FAO-56 Eq. 62 (Kc_mid) / Eq. 65 (Kc_end) climate correction.
    Only applied for Kc >= 0.45 (per FAO-56, and for Kc_end only when the
    tabulated value is > 0.45). rhmin_mean is the mean RHmin (%) over the
    stage; h_m the mean plant height for that stage.
    """
    if stage == "mid" or (stage == "end" and kc_tab > 0.45):
        return (kc_tab
                + (0.04 * (u2 - 2.0) - 0.004 * (rhmin_mean - 45.0))
                * (h_m / 3.0) ** 0.3)
    return kc_tab


def build_kc_series(dates, tmax, tmin, crop, plant_day, u2,
                    stage_lengths=None, kc_values=None, height=None):
    """
    Build a daily Kc series over the full record for a repeating annual
    growing season starting at plant_day each year. Returns a Series.
    """
    if crop is not None:
        (Li, Ld, Lm, Ll, kc_ini, kc_mid, kc_end, h) = CROP_TABLE[crop]
    else:
        Li = Ld = Lm = Ll = None
        kc_ini = kc_mid = kc_end = None
        h = 1.0
    # CLI overrides
    if stage_lengths is not None:
        Li, Ld, Lm, Ll = stage_lengths
    if kc_values is not None:
        kc_ini, kc_mid, kc_end = kc_values
    if height is not None:
        h = height
    if None in (Li, Ld, Lm, Ll, kc_ini, kc_mid, kc_end):
        sys.exit("ERROR: crop undefined; pass --crop or "
                 "--kc-stages and --kc-values")

    season_len = Li + Ld + Lm + Ll

    # RHmin(%) from Tmin dewpoint proxy: 100 * e0(Tmin)/e0(Tmax).
    tmx = np.asarray(tmax, dtype=float)
    tmn = np.asarray(tmin, dtype=float)
    rhmin = 100.0 * sat_vapour_pressure(tmn) / sat_vapour_pressure(tmx)
    rhmin = np.clip(rhmin, 5.0, 100.0)
    rhmin_ser = pd.Series(rhmin, index=dates)

    doy = np.asarray(dates.dayofyear, dtype=float)
    # Day-into-season for each date (0 at plant_day), wrapping the year.
    dis = (doy - plant_day) % 365.0

    # Mean RHmin over mid and late windows (for Eq. 62/65), approximated
    # by the record-wide mean RHmin on days that fall in each stage.
    in_mid = (dis >= Li + Ld) & (dis < Li + Ld + Lm)
    in_late = (dis >= Li + Ld + Lm) & (dis < season_len)
    rhmin_mid = float(rhmin_ser[in_mid].mean()) if in_mid.any() else 45.0
    rhmin_late = float(rhmin_ser[in_late].mean()) if in_late.any() else 45.0

    kc_mid_adj = climate_adjust_kc(kc_mid, h, rhmin_mid, u2, "mid")
    kc_end_adj = climate_adjust_kc(kc_end, h, rhmin_late, u2, "end")

    kc = np.full(len(dates), KC_OFF, dtype=float)
    for k, d in enumerate(dis):
        if d < Li:                                   # initial
            kc[k] = kc_ini
        elif d < Li + Ld:                            # development ramp
            f = (d - Li) / Ld
            kc[k] = kc_ini + f * (kc_mid_adj - kc_ini)
        elif d < Li + Ld + Lm:                       # mid-season
            kc[k] = kc_mid_adj
        elif d < season_len:                         # late-season ramp
            f = (d - (Li + Ld + Lm)) / Ll
            kc[k] = kc_mid_adj + f * (kc_end_adj - kc_mid_adj)
        # else: off-season -> KC_OFF (already set)
    return pd.Series(kc, index=dates)


# ----------------------------------------------------------------------
# Snow module (temperature-threshold + degree-day melt)
# ----------------------------------------------------------------------
def snow_step(precip, tmean, snowpack, snow_params=None):
    """Return (effective_water_to_soil, new_snowpack)."""
    if snow_params is None:
        snow_params = SNOW
    if tmean <= snow_params["t_snow"]:
        snowpack += precip            # all precip accumulates as snow
        rain = 0.0
    else:
        rain = precip                 # all precip is rain
    melt = 0.0
    if tmean > snow_params["t_melt"] and snowpack > 0.0:
        melt = min(snowpack, snow_params["melt_factor"] * (tmean - snow_params["t_melt"]))
        snowpack -= melt
    return rain + melt, snowpack


# ----------------------------------------------------------------------
# Core leaky-bucket update
# ----------------------------------------------------------------------
def beta(w, wmax=WMAX):
    """Evaporation efficiency = w / wmax (linearly water-limited)."""
    return w / wmax


def runoff(w, peff, p, wmax=WMAX):
    """
    CPC-style runoff: surface runoff scales with the storage ratio raised
    to Bm times the incoming water, plus a linear base runoff.
    """
    ratio = max(0.0, min(1.0, w / wmax))
    r_surf = p["alpha_surf"] * (ratio ** p["Bm"]) * peff
    r_base = p["alpha_base"] * w * ratio
    return r_surf + r_base


# ----------------------------------------------------------------------
# Irrigation (FAO-56 deficit-refill, single most-recent event)
#
# Irrigation enters the water balance as an effective water input on the
# day it was applied, exactly like precipitation (pyfao56; Allen et al.
# 1998, Ch. 8). The event is placed `daysbefore` days before the end of
# the record and propagated forward by the normal bucket dynamics.
#
# Depth: FAO deficit-refill. The depth needed to return the root zone to
# field capacity equals the current depletion,  D = WMAX - w(t). In
# FAO-56 terms irrigation replenishes the depletion Dr back to Dr = 0
# (field capacity). In this one-layer bucket, field capacity = WMAX.
# Applied at 100% efficiency (no method/application-loss distinction).
# ----------------------------------------------------------------------


def run_model(precip, tmean, pe, lat, params=PARAMS, snow=True,
              w0_frac=0.5, spinup_years=1, irrig_daysbefore=None,
              wmax=WMAX, snow_params=None):
    """Run the daily water balance. Returns a DataFrame of states/fluxes."""
    if snow_params is None:
        snow_params = SNOW
    idx = precip.index
    n = len(idx)
    w = w0_frac * wmax
    snowpack = 0.0

    # Single irrigation event: index of the day it was applied. The event
    # is `irrig_daysbefore` days before the last record. Deficit-refill
    # depth is computed in-loop from w(t).
    irrig_k = None
    if irrig_daysbefore is not None:
        if irrig_daysbefore < 0 or irrig_daysbefore >= n:
            sys.exit(f"ERROR: --daysbefore must be in [0, {n-1}]")
        irrig_k = n - 1 - irrig_daysbefore

    out = np.zeros((n, 7))  # peff, snow, w, E, R, G, irrig
    pv = precip.values
    tv = tmean.values
    ev = pe.values

    for k in range(n):
        p = pv[k]
        t = tv[k]
        if snow:
            peff, snowpack = snow_step(p if not np.isnan(p) else 0.0, t,
                                       snowpack, snow_params=snow_params)
        else:
            peff = p if not np.isnan(p) else 0.0

        pe_k = ev[k] if not np.isnan(ev[k]) else 0.0

        # --- Irrigation (single deficit-refill event) ---
        # Depth to bring storage back to field capacity (= WMAX), applied
        # at 100% efficiency.
        irrig = 0.0
        if irrig_k is not None and k == irrig_k:
            irrig = max(0.0, wmax - w)
            peff += irrig

        # Fluxes evaluated on the storage at start of step (explicit).
        E = beta(w, wmax=wmax) * pe_k
        R = runoff(w, peff, params, wmax=wmax)
        G = params["gamma"] * (w / wmax) # wait, the original was G = params["gamma"] * (w / WMAX) - but let's change to wmax since we are refactoring. Wait, is it w/WMAX or w/wmax? It's the storage fraction so it should be w/wmax.
        G = params["gamma"] * (w / wmax)

        w_new = w + peff - E - R - G

        # Enforce bounds; spill overflow into runoff, deficit into reduced E.
        if w_new > wmax:
            R += (w_new - wmax)
            w_new = wmax
        if w_new < 0.0:
            # scale losses back so storage hits exactly zero
            deficit = -w_new
            total_loss = E + R + G
            if total_loss > 0:
                E -= E / total_loss * deficit
                R -= R / total_loss * deficit
                G -= G / total_loss * deficit
            w_new = 0.0

        out[k] = (peff, snowpack, w_new, E, R, G, irrig)
        w = w_new

    df = pd.DataFrame(
        out, index=idx,
        columns=["P_eff", "snowpack", "w", "E", "R", "G", "irrig"],
    )
    df["P_obs"] = precip.values
    df["Tmean"] = tmean.values
    df["PE"] = pe.values
    df["w_frac"] = df["w"] / wmax

    # Drop spin-up.
    if spinup_years > 0:
        cutoff = idx[0] + pd.DateOffset(years=spinup_years)
        df = df[df.index >= cutoff]
    return df[["P_obs", "Tmean", "PE", "P_eff", "irrig", "snowpack",
               "w", "E", "R", "G", "w_frac"]]


# ----------------------------------------------------------------------
def cpc_leaky_bucket_pipeline(
    base=None,
    pcp=None,
    tmax=None,
    tmin=None,
    lat=None,
    elev=None,
    out="soil_moisture.csv",
    no_snow=False,
    w0=0.5,
    spinup=1,
    crop=None,
    plant_day=1,
    kc_stages=None,
    kc_values=None,
    crop_height=None,
    elev_file=None,
    lon=None,
    daysbefore=None,
):
    if base and not (pcp and tmax and tmin):
        pcp, tmax, tmin = derive_paths(base)
    if not (pcp and tmax and tmin):
        sys.exit("ERROR: supply --base, or all of --pcp --tmax --tmin")

    if lat is None:
        lat = lat_from_filename(pcp)
        if lat is None:
            sys.exit("ERROR: could not parse latitude; pass --lat")

    # Resolve elevation: explicit --elev wins; else look up in --elev-file
    # by the point's lat/lon; else default to 0 m.
    if elev is None:
        if elev_file:
            if lon is None:
                _, lon = latlon_from_filename(pcp)
                if lon is None:
                    sys.exit("ERROR: --elev-file needs longitude; pass --lon")
            table, elat, elon, eelev = load_elevation_file(elev_file)
            elev = lookup_elevation(lat, lon,
                                    table, elat, elon, eelev)
        else:
            elev = 0.0

    pcp_series = read_point_file(pcp)
    tmax_series = read_point_file(tmax)
    tmin_series = read_point_file(tmin)

    # Align on common dates.
    df = pd.concat({"pcp": pcp_series, "tmax": tmax_series, "tmin": tmin_series}, axis=1).dropna(
        subset=["tmax", "tmin"])
    df["pcp"] = df["pcp"].fillna(0.0)
    tmean = (df["tmax"] + df["tmin"]) / 2.0

    pe = penman_monteith_eto(df["tmax"], df["tmin"], df.index,
                             lat, elev)

    # FAO-56 crop coefficient: ETc = Kc * ETo. Applied only if requested
    # via --crop or explicit --kc-values overrides; otherwise Kc = 1
    # (bare reference ET, unchanged from before).
    if crop is not None or kc_values is not None:
        kc = build_kc_series(
            df.index, df["tmax"], df["tmin"],
            crop=crop, plant_day=plant_day, u2=PM["u2"],
            stage_lengths=kc_stages, kc_values=kc_values,
            height=crop_height,
        )
        pe = pe * kc          # ETc = Kc * ETo

    result = run_model(
        df["pcp"], tmean, pe, lat,
        snow=not no_snow, w0_frac=w0, spinup_years=spinup,
        irrig_daysbefore=daysbefore,
    )
    if out:
        result.to_csv(out, float_format="%.3f",
                      index_label="date")
    return result


def main():
    ap = argparse.ArgumentParser(description="CPC leaky-bucket soil moisture")
    ap.add_argument("--base", help="one RT_* point file; others derived")
    ap.add_argument("--pcp")
    ap.add_argument("--tmax")
    ap.add_argument("--tmin")
    ap.add_argument("--lat", type=float)
    ap.add_argument("--elev", type=float, default=None,
                    help="site elevation (m) for FAO-56 PM pressure/gamma; "
                         "overrides --elev-file. Default 0 m if neither given")
    ap.add_argument("--out", default="soil_moisture.csv")
    ap.add_argument("--no-snow", action="store_true")
    ap.add_argument("--w0", type=float, default=0.5,
                    help="initial storage as fraction of wmax")
    ap.add_argument("--spinup", type=int, default=1,
                    help="spin-up years to discard")
    # --- FAO-56 crop coefficient (ETc = Kc * ETo) ---
    ap.add_argument("--crop", choices=sorted(CROP_TABLE.keys()),
                    help="apply FAO-56 seasonal Kc for this crop; "
                         "omit for bare reference ET (Kc=1)")
    ap.add_argument("--plant-day", type=int, default=1,
                    help="growing-season start day-of-year (repeats yearly)")
    ap.add_argument("--kc-stages", type=int, nargs=4,
                    metavar=("LINI", "LDEV", "LMID", "LLATE"),
                    help="override stage lengths (days)")
    ap.add_argument("--kc-values", type=float, nargs=3,
                    metavar=("KCINI", "KCMID", "KCEND"),
                    help="override tabulated Kc_ini/mid/end")
    ap.add_argument("--crop-height", type=float,
                    help="override crop height (m) for Kc climate adjustment")
    ap.add_argument("--elev-file",
                    help="'lon lat elev' grid file; elevation looked up by "
                         "the point's lat/lon (overridden by explicit --elev)")
    ap.add_argument("--lon", type=float,
                    help="point longitude for --elev-file lookup "
                         "(parsed from filename if omitted)")
    # --- Irrigation (FAO-56 deficit-refill, single most-recent event) ---
    ap.add_argument("--daysbefore", type=int,
                    help="refill root zone to field capacity this many days "
                         "before the end of record (0 = last day)")
    args = ap.parse_args()

    cpc_leaky_bucket_pipeline(
        base=args.base,
        pcp=args.pcp,
        tmax=args.tmax,
        tmin=args.tmin,
        lat=args.lat,
        elev=args.elev,
        out=args.out,
        no_snow=args.no_snow,
        w0=args.w0,
        spinup=args.spinup,
        crop=args.crop,
        plant_day=args.plant_day,
        kc_stages=args.kc_stages,
        kc_values=args.kc_values,
        crop_height=args.crop_height,
        elev_file=args.elev_file,
        lon=args.lon,
        daysbefore=args.daysbefore,
    )


if __name__ == "__main__":
    main()
