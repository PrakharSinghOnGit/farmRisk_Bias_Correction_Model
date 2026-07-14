---
title: FarmRisk
emoji: 🏆
colorFrom: red
colorTo: pink
sdk: docker
pinned: false
license: mit
short_description: FarmRisk forecast model
---

# Hydrology & Weather Forecast API Server

This Hugging Face Space hosts a FastAPI server exposing CPC soil moisture simulation and IDW XGBoost bias-corrected weather forecast pipelines.

## API Documentation

When running, the interactive Swagger documentation is available at `/docs` (or `/redoc`).

### Endpoints

#### 1. CPC Leaky-Bucket Soil Moisture

Returns the last 365 daily records of simulated soil moisture, precipitation, evapotranspiration, and climatological soil-moisture percentiles.

- **URL**: `/moisture`
- **Method**: `GET`
- **Query Parameters**:
  - `lat` (float, required): Latitude of target location.
  - `lon` (float, required): Longitude of target location.
  - `daysbefore` (int, optional): Number of days before today for deficit-refill irrigation.
- **Example Query**:
  ```bash
  curl "https://<your-space-url>/moisture?lat=28.6139&lon=77.2090"
  ```

#### 2. Bias-Corrected Weather Forecast

Returns 16 days of IDW-weighted bias-corrected weather forecast data.

- **URL**: `/forecast`
- **Method**: `GET`
- **Query Parameters**:
  - `lat` (float, required): Latitude of target location.
  - `lon` (float, required): Longitude of target location.
- **Example Query**:
  ```bash
  curl "https://<your-space-url>/forecast?lat=28.6139&lon=77.2090"
  ```

#### 3. Execution Statistics

Returns the execution count, last run duration, and rolling average duration for the pipelines. This is useful for displaying dynamic progress/loading bars in client UIs.

- **URL**: `/stats`
- **Method**: `GET`
- **Example Query**:
  ```bash
  curl "https://<your-space-url>/stats"
  ```
