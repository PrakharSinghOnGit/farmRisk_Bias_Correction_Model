---
title: FarmRisk Weather and Soil Moisture Forecast API
emoji: none
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# FarmRisk: Weather and Soil Moisture Forecast API

A FastAPI backend hosting an integrated weather forecast bias-correction model and a derived soil moisture hydrology model. 

---

## 1. Scientific and Modeling Architecture

This API runs two main scientific pipelines sequentially: a machine learning bias-correction model for weather forecasts, and a physical water-balance model for soil moisture.

```
                  [Live Open-Meteo Forecast]
                               │
                               ▼
   [XGBoost Bias Correction (Per-Grid Point 0.25°)] ◄─── [ERA5 & IMD Historical Data]
                               │
                               ▼
            [Bias-Corrected Weather Forecast]
                               │
                               ▼
      [CPC Leaky-Bucket Soil Moisture Model] ◄────────── [Calibration & Checkpoints]
                               │
                               ▼
       [Combined Weather & Soil Moisture Forecast]
```

### Weather Forecast Bias-Correction (XGBoost)
Raw numerical weather prediction (NWP) forecasts from Open-Meteo are subject to systemic biases depending on geography, elevation, and local topography. The bias-correction model uses **Extreme Gradient Boosting (XGBoost)** to map forecasted parameters to historically observed ground-truth values.

* **Target Variables:** Maximum Temperature (tmax), Minimum Temperature (tmin), and Daily Precipitation Sum (pcp).
* **Ground Truth:** India Meteorological Department (IMD) high-resolution gridded daily datasets (0.25° resolution).
* **Reference Historical Dataset:** ERA5 reanalysis data.
* **Features:** 7-day rolling statistics of raw forecasts, grid elevation (from SRTM 90m), and spatial distance weights.
* **Grid-Level Cache:** Models are trained on-the-fly for the specific 0.25° grid cell corresponding to the requested latitude/longitude. Once trained, the model is cached to disk under `models/models_cache_pergrid` to avoid retraining for neighboring villages within the same grid cell. Cached models expire after 7 days to account for seasonal adjustments.

### Soil Moisture Simulation (CPC Leaky-Bucket Model)
Derived soil moisture is simulated using a physical **Climate Prediction Center (CPC) Leaky-Bucket Water Balance Model**, which operates as a Markov recursion state system:

$$w(t+1) = f(w(t), \text{snowpack}(t), \text{forcing}(t+1))$$

* **Model Inputs:** Daily temperature and daily precipitation from the bias-corrected weather forecast.
* **Potential Evapotranspiration (PE):** Computed using the Thornthwaite formulation, which relies on calendar-month temperatures.
* **Calibration Parameters:** Sourced from `master_calibration_1D.csv` (saturated soil capacity, runoff parameter, and evapotranspiration coefficients).
* **Climatological Percentiles:** Computed relative to a historical window of observations (1990–present) to identify dry or wet soil moisture anomalies on any given day.
* **Incremental Checkpointing:** To avoid replaying 35+ years of historical day-by-day calculations on every API call, the API persists a village-specific state checkpoint under `models/soil_moisture_checkpoints/`. When a new request arrives, the system loads the checkpoint, simulates the small number of newly observed days, updates the checkpoint state, and calculates the forecast.

---

## 2. API Endpoints

### POST `/village-report`
Primary API entry point. Takes coordinates and returns the bias-corrected weather forecast alongside the derived soil moisture forecast.
* **Request Body:**
  ```json
  {
    "lat": 22.3,
    "lon": 72.6,
    "village_id": 12345
  }
  ```
* **Response Output:** Returns bias-corrected weather forecast datasets, derived daily soil moisture storage ($w$), soil moisture percentiles, and runtime performance metrics.

### POST `/forecast`
Isolated debugging endpoint returning only the bias-corrected weather forecast (no soil moisture calculations).

### GET `/health`
Returns the status of the service, including cache health, missing file reports, and running execution stats:
* **`average_runtime_seconds`**: Running average of successful inference times.
* **`inference_count`**: Number of successful calculations performed.

---

## 3. Runtime Statistics Tracking

To track system performance without file growth or memory leakage, the API implements an online **incremental averaging algorithm** to record inference runtimes in `models/avgRunTime.json`. 

Each successful computation updates the average via:

$$\mu_{n+1} = \mu_n + \frac{x - \mu_n}{n + 1}$$

Where:
* $\mu_n$ is the previous average runtime (`average`).
* $x$ is the runtime of the current computation.
* $n$ is the previous inference count (`count`).

This ensures that the metric is computed dynamically, safely under a process-level thread lock, and without storing individual inference history.

---

## 4. Git LFS Setup for Datasets

Large dataset files (parquet and calibration CSVs) total approximately 1.78 GB. They must be managed via **Git Large File Storage (LFS)**.

```bash
# Initialize LFS in repository
git lfs install

# Configure file types for LFS tracking
git lfs track "*.parquet"
git lfs track "*.csv"

# Add tracking metadata
git add .gitattributes
```

---

## 5. Local Docker Deployment & Verification

### Build the Image
```bash
docker build -t forecast-model-api .
```

### Run the Container
```bash
docker run -p 8000:7860 forecast-model-api
```

### Test Output Generation
Query the `/health` endpoint or send an inference payload:
```bash
curl -X POST "http://localhost:8000/village-report" \
     -H "Content-Type: application/json" \
     -d '{"lat": 22.3, "lon": 72.6, "village_id": 12345}'
```

---

## 6. Hugging Face Spaces Deployment

To deploy this backend directly to Hugging Face Spaces:
1. Create a new Space on Hugging Face. Select **Docker** as the SDK.
2. Add your Space repository URL as a git remote:
   ```bash
   git remote add hf https://huggingface.co/spaces/YOUR_HF_USERNAME/YOUR_SPACE_NAME
   ```
3. Push to Hugging Face:
   ```bash
   git push hf main
   ```
   *(Use your Hugging Face Access Token with write privileges when prompted for your password)*
