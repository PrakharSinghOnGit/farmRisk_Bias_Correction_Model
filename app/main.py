import os
import sys
import json
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response

# Add current workspace directory to sys.path if not present
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.soilMoisture import run_soil_moisture_pipeline
from app.forcast import run_forecast_pipeline
from app.stats import STATS_FILE

app = FastAPI(
    title="Hydrology and Weather API Server",
    description="API endpoints for Soil Moisture and Weather Forecast models.",
    version="1.0.0"
)


@app.get("/moisture", summary="Get Soil Moisture Data")
def get_moisture(
    lat: float = Query(..., description="Latitude of the location"),
    lon: float = Query(..., description="Longitude of the location"),
    daysbefore: int = Query(None, description="Days before for irrigation refill (optional)")
):
    """
    Simulates the CPC leaky-bucket soil moisture model for a given coordinate.
    Returns the last 365 days of simulation data (e.g., date, soil moisture w, precipitation, evapotranspiration, percentiles, etc.) in JSON format.
    """
    try:
        res_json = run_soil_moisture_pipeline(
            lat=lat,
            lon=lon,
            daysbefore=daysbefore,
            save_csv=False
        )
        if not res_json:
            raise HTTPException(status_code=404, detail="No simulation results generated.")
        return Response(content=res_json, media_type="application/json")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Soil moisture simulation error: {str(e)}")


@app.get("/forecast", summary="Get Weather Forecast Data")
def get_forecast(
    lat: float = Query(..., description="Latitude of the location"),
    lon: float = Query(..., description="Longitude of the location")
):
    """
    Fetches Open-Meteo weather data and applies IDW XGBoost bias correction.
    Returns the 16-day weather forecast (including date, max/min temperatures, and precipitation) in JSON format.
    """
    try:
        res_json = run_forecast_pipeline(
            lat=lat,
            lon=lon
        )
        if not res_json:
            raise HTTPException(status_code=404, detail="No forecast results generated.")
        return Response(content=res_json, media_type="application/json")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Weather forecast pipeline error: {str(e)}")


@app.get("/stats", summary="Get Pipeline Run Statistics")
def get_stats():
    """
    Returns the run count, last run duration, and average run duration for both
    forecast and soil moisture pipelines.
    """
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading stats: {str(e)}")
    return {
        "forecast": {"count": 0, "last_time": 0.0, "avg_time": 0.0},
        "soil_moisture": {"count": 0, "last_time": 0.0, "avg_time": 0.0}
    }


@app.get("/health", summary="Get System and Data Health")
def get_health():
    """
    Checks environment details, directory permissions, and validates that all 
    required data files (elevation, forecast history, and IMD observation parquets) 
    are present and readable on the filesystem.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    required_files = {
        "grid_elevation.parquet": os.path.join(base_dir, "data", "grid_elevation.parquet"),
        "om_forecast_all.parquet": os.path.join(base_dir, "data", "om_forecast_all.parquet"),
        "imd_tmax_daily.parquet": os.path.join(base_dir, "data", "IMD_parquets", "imd_tmax_daily.parquet"),
        "imd_tmin_daily.parquet": os.path.join(base_dir, "data", "IMD_parquets", "imd_tmin_daily.parquet"),
        "imd_pcp_daily.parquet": os.path.join(base_dir, "data", "IMD_parquets", "imd_pcp_daily.parquet"),
        "master_calibration_1D.csv": os.path.join(base_dir, "data", "master_calibration_1D.csv")
    }

    health_status = {}
    all_fine = True

    for name, path in required_files.items():
        exists = os.path.exists(path)
        is_file = os.path.isfile(path)
        readable = os.access(path, os.R_OK) if exists else False
        
        health_status[name] = {
            "exists": exists,
            "is_file": is_file,
            "readable": readable,
            "size_bytes": os.path.getsize(path) if exists else 0,
            "resolved_path": path
        }
        if not (exists and readable):
            all_fine = False

    # Also check if stats.json is present/writable
    stats_path = os.path.join(base_dir, "stats.json")
    stats_exists = os.path.exists(stats_path)
    stats_writable = os.access(stats_path, os.W_OK) if stats_exists else os.access(base_dir, os.W_OK)
    
    health_status["stats.json"] = {
        "exists": stats_exists,
        "writable": stats_writable,
        "resolved_path": stats_path
    }
    
    status = "ok" if all_fine else "unhealthy"

    return {
        "status": status,
        "environment": {
            "PORT": os.getenv("PORT", "7860 (default)"),
            "PYTHON_VERSION": sys.version,
            "CWD": os.getcwd()
        },
        "data_files_health": health_status
    }


if __name__ == "__main__":
    import uvicorn
    # Use PORT environment variable (default: 7860 for Hugging Face Spaces)
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
