import logging
import os
import json
import threading
import time
from datetime import date
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.forcastModel import (
    get_village_forecast,
    surrounding_grid_points,
    clear_model_cache,
    ERA5_DIR as FM_ERA5_DIR,
    FORECAST_PARQUET as FM_FORECAST_PARQUET,
    ELEV_PARQUET as FM_ELEV_PARQUET,
)

# NOTE: if your soil-moisture file is saved under a different name, update
# this import to match (kept as a separate module, same pattern as forcastModel).
from app.soilMoistureModel import (
    compute_village_soil_moisture,
    clear_calibration_cache,
    IMD_PARQUET as SM_IMD_PARQUET,
    CALIB_FILE as SM_CALIB_FILE,
    CHECKPOINT_DIR as SM_CHECKPOINT_DIR,
)

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "village_forecasts_pergrid"
OPEN_METEO_API_KEY = os.getenv("OPEN_METEO_API_KEY", "")
CACHE_CLEAR_KEY = os.getenv("CACHE_CLEAR_KEY", "")
LOG_MODEL_OUTPUT = os.getenv("LOG_MODEL_OUTPUT", "false").lower() == "true"

# Serializes the forecast+soil-moisture pipeline. Keep this at 1 unless the
# hosting machine has CPU/RAM to spare -- a cold grid (forecast) or a cold
# village (soil moisture) can still be a real one-time cost even though warm
# requests are now cheap lookups. Raise this once your caches are mostly warm.
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_FORECASTS", "1"))

logger = logging.getLogger("farmrisk.village_api")


class ForecastRequest(BaseModel):
    lat: float = Field(..., ge=6.0, le=38.0, description="Latitude in degrees")
    lon: float = Field(..., ge=66.0, le=99.0, description="Longitude in degrees")
    village_id: Optional[int] = Field(
        None, description="Stable village id, used to key the soil-moisture "
                          "checkpoint. Falls back to lat/lon if omitted."
    )


class CacheClearRequest(BaseModel):
    key: str = Field(..., min_length=1)


class RuntimeFile(BaseModel):
    path: str
    exists: bool
    size_mb: float | None = None


app = FastAPI(
    title="FarmRisk Village Forecast + Soil Moisture API",
    version="2.0.0",
    description=(
        "Single-call HTTP API: takes lat/lon, returns a bias-corrected weather "
        "forecast AND the derived soil-moisture forecast together."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline_lock = threading.Semaphore(MAX_CONCURRENT_REQUESTS)
# Guards the shared response cache dict itself (not the actual computation --
# holding this during a slow forecast/soil-moisture run would serialize the
# whole API even below MAX_CONCURRENT_REQUESTS). A rare double-compute race
# on a cache miss is possible under concurrency > 1; it's harmless and
# idempotent, just occasionally wasted work, which is the right tradeoff
# versus locking around the entire request.
_cache_lock = threading.Lock()
response_cache: dict[str, dict[str, Any]] = {}
response_cache_day = date.today().isoformat()


_runtime_file_lock = threading.Lock()


def update_avg_runtime(x: float) -> None:
    """Updates the running average runtime in models/avgRunTime.json using incremental averaging."""
    models_dir = Path(SM_CHECKPOINT_DIR).parent
    file_path = models_dir / "avgRunTime.json"
    
    with _runtime_file_lock:
        old_average = 0.0
        count = 0
        if file_path.exists():
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    old_average = float(data.get("average", 0.0))
                    count = int(data.get("count", 0))
            except Exception:
                pass
        
        new_average = old_average + (x - old_average) / (count + 1)
        new_count = count + 1
        
        try:
            with open(file_path, "w") as f:
                json.dump({
                    "average": round(new_average, 6),
                    "count": new_count,
                    "last_runtime": round(x, 6)
                }, f, indent=2)
        except Exception as e:
            logger.error("Failed to write avgRunTime.json: %s", e)


def _file_info(path: Path) -> RuntimeFile:
    exists = path.exists()
    return RuntimeFile(
        path=str(path),
        exists=exists,
        size_mb=round(path.stat().st_size / 1024 / 1024, 2) if exists else None,
    )


def required_files() -> list[Path]:
    """
    Pulled directly from the modules' own path constants rather than
    re-declared here, so this can never silently drift out of sync with
    what forcastModel.py / soilMoistureModel.py actually read from.
    """
    files = [
        BASE_DIR / "forcastModel.py",
        BASE_DIR / "soilMoistureModel.py",
        Path(FM_ERA5_DIR) / "training_tmax_daily.parquet",
        Path(FM_ERA5_DIR) / "training_tmin_daily.parquet",
        Path(FM_ERA5_DIR) / "training_pcp_daily.parquet",
        Path(FM_FORECAST_PARQUET),
        Path(FM_ELEV_PARQUET),
        Path(SM_CALIB_FILE),
    ]
    files.extend(Path(p) for p in SM_IMD_PARQUET.values())
    return files


def missing_files() -> list[str]:
    return [str(path) for path in required_files() if not path.exists()]


# ==============================================================================
# Response cache — keyed by exact point + day, NOT by grid block.
#
# The original cached by which 4 grid points surround the location. That's
# wrong: two villages sharing a 0.25 grid cell would get back identical
# elevation, identical IDW distance weights, and identical live Open-Meteo
# values -- all of which are point-specific, not cell-specific. This caches
# the exact (lat, lon, day) instead, so it only ever short-circuits a
# literal repeat call for the same point (e.g. a frontend retry), never a
# different village nearby.
#
# Note this cache is now mostly about saving the elevation + Open-Meteo
# network round-trips for repeat calls -- the expensive part (training a
# grid's model) is already cached at the grid level inside forcastModel.py,
# so this layer isn't load-bearing for correctness or for avoiding retrains.
# ==============================================================================

def _refresh_daily_cache() -> None:
    global response_cache_day
    today = date.today().isoformat()
    with _cache_lock:
        if response_cache_day != today:
            response_cache.clear()
            response_cache_day = today


def _point_cache_key(lat: float, lon: float, village_id: Optional[int]) -> str:
    village_part = f"vid:{village_id}" if village_id is not None else f"{lat:.4f},{lon:.4f}"
    return f"{response_cache_day}:{village_part}"


def _cache_get(key: str) -> Optional[dict[str, Any]]:
    with _cache_lock:
        cached = response_cache.get(key)
    return dict(cached) if cached is not None else None


def _cache_set(key: str, value: dict[str, Any]) -> None:
    with _cache_lock:
        response_cache[key] = value


# ==============================================================================
# Core pipeline
# ==============================================================================

def _run_forecast(lat: float, lon: float) -> dict[str, Any]:
    missing = missing_files()
    if missing:
        raise RuntimeError("Missing required model files: " + ", ".join(missing))

    kwargs = {}
    if OPEN_METEO_API_KEY:
        kwargs["apikey"] = OPEN_METEO_API_KEY

    res = get_village_forecast(lat, lon, **kwargs)
    if not res.get("success", False):
        error_msg = res.get("error", {}).get("message", "Forecast model returned no result")
        raise RuntimeError(f"Forecast model error: {error_msg}")
    return res


def _run_village_report(lat: float, lon: float, village_id: Optional[int]) -> dict[str, Any]:
    """The full pipeline: forecast, then soil moisture derived from it."""
    started = time.time()

    forecast = _run_forecast(lat, lon)
    forecast["runtime_seconds"] = round(time.time() - started, 4)

    sm_started = time.time()
    soil_moisture = compute_village_soil_moisture(
        lat, lon, forecast.get("forecast", []), village_id=village_id,
    )
    soil_moisture["runtime_seconds"] = round(time.time() - sm_started, 4)

    return {
        "requested_lat": lat,
        "requested_lon": lon,
        "village_id": village_id,
        "forecast": forecast,
        "soil_moisture": soil_moisture,
        "cache_hit": False,
        "total_runtime_seconds": round(time.time() - started, 4),
    }


def get_village_report_cached(lat: float, lon: float, village_id: Optional[int]) -> dict[str, Any]:
    _refresh_daily_cache()
    key = _point_cache_key(lat, lon, village_id)

    cached = _cache_get(key)
    if cached is not None:
        cached["cache_hit"] = True
        cached["cache_key"] = key
        return cached

    response = _run_village_report(lat, lon, village_id)
    response["cache_key"] = key
    _cache_set(key, response)

    update_avg_runtime(response["total_runtime_seconds"])

    logger.info(
        "village report lat=%s lon=%s village_id=%s seconds=%s "
        "forecast_cold_start=%s soil_moisture_cold_start=%s",
        lat, lon, village_id, response["total_runtime_seconds"],
        response["forecast"].get("cold_start"),
        response["soil_moisture"].get("cold_start"),
    )
    return response


# ==============================================================================
# Endpoints
# ==============================================================================

@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "FarmRisk Village Forecast + Soil Moisture API",
        "endpoints": {
            "combined": "POST /village-report",
            "forecast_only": "POST /forecast",
            "clear_cache": "POST /cache/clear",
            "health": "GET /health",
        },
    }


@app.get("/health")
def health() -> dict[str, Any]:
    missing = missing_files()
    checkpoint_count = 0
    if os.path.isdir(SM_CHECKPOINT_DIR):
        checkpoint_count = len(os.listdir(SM_CHECKPOINT_DIR))
    
    avg_runtime = 0.0
    inference_count = 0
    avg_runtime_file = Path(SM_CHECKPOINT_DIR).parent / "avgRunTime.json"
    if avg_runtime_file.exists():
        try:
            with open(avg_runtime_file, "r") as f:
                data = json.load(f)
                avg_runtime = float(data.get("average", 0.0))
                inference_count = int(data.get("count", 0))
        except Exception:
            pass

    return {
        "ok": not missing,
        "base_dir": str(BASE_DIR),
        "open_meteo_key_configured": bool(OPEN_METEO_API_KEY),
        "max_concurrent_requests": MAX_CONCURRENT_REQUESTS,
        "response_cache_day": response_cache_day,
        "response_cache_entries": len(response_cache),
        "soil_moisture_checkpoints": checkpoint_count,
        "average_runtime_seconds": avg_runtime,
        "inference_count": inference_count,
        "missing_files": missing,
    }


@app.get("/runtime-files", response_model=list[RuntimeFile])
def runtime_files() -> list[RuntimeFile]:
    return [_file_info(path) for path in required_files()]


@app.post("/village-report")
async def village_report(payload: ForecastRequest) -> dict[str, Any]:
    """Single call: bias-corrected forecast + derived soil moisture, together."""
    with pipeline_lock:
        try:
            return await run_in_threadpool(
                get_village_report_cached, payload.lat, payload.lon, payload.village_id,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/forecast")
async def forecast_post(payload: ForecastRequest) -> dict[str, Any]:
    """Forecast only, no soil moisture -- kept for isolated testing/debugging."""
    started = time.time()
    with pipeline_lock:
        try:
            res = await run_in_threadpool(_run_forecast, payload.lat, payload.lon)
            elapsed = round(time.time() - started, 4)
            update_avg_runtime(elapsed)
            return res
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/cache/clear")
def clear_cache(payload: CacheClearRequest) -> dict[str, Any]:
    """
    Clears in-memory response cache + in-memory model/calibration caches only.
    Deliberately does NOT touch the on-disk trained-model cache or the
    soil-moisture checkpoints -- those hold real, expensive-to-rebuild state
    and shouldn't be one key away from being wiped by accident.
    """
    if not CACHE_CLEAR_KEY:
        raise HTTPException(
            status_code=503,
            detail="CACHE_CLEAR_KEY is not configured on the server",
        )
    if payload.key != CACHE_CLEAR_KEY:
        raise HTTPException(status_code=403, detail="Invalid cache clear key")

    with _cache_lock:
        cleared = len(response_cache)
        response_cache.clear()

    cleared_models = clear_model_cache()
    clear_calibration_cache()

    return {
        "ok": True,
        "cleared_response_cache_entries": cleared,
        "cleared_model_cache_entries": cleared_models,
        "response_cache_day": response_cache_day,
        "soil_moisture_checkpoints_untouched": True,
        "disk_model_cache_untouched": True,
    }