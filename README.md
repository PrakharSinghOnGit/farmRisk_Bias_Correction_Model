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
  - `crop` (string, optional): Crop type for FAO-56 Kc application.
  - `daysbefore` (int, optional): Number of days before today for deficit-refill irrigation.
- **Example Query**:
  ```bash
  curl "https://<your-space-url>/moisture?lat=28.6139&lon=77.2090&crop=maize&daysbefore=7"
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

#### 4. System and Data Health

Checks environment details, directory permissions, and validates that all required data files (elevation, forecast history, and IMD observation parquets) are present and readable on the filesystem.

- **URL**: `/health`
- **Method**: `GET`
- **Example Query**:
  ```bash
  curl "https://<your-space-url>/health"
  ```

---

## Dataset & Git LFS Setup (GitHub)

The `data/` directory contains large model and observation parquet datasets which exceed GitHub's standard file size limits. To handle this, the repository uses **Git LFS (Large File Storage)** on GitHub.

The parquet files inside `data/` are tracked using Git LFS (configured in `.gitattributes`).

### How to Clone and Download the Data (with Git LFS)

When cloning this repository, you must have Git LFS installed on your system to download the actual dataset files instead of pointers.

1. **Install Git LFS** (if not already installed):
   - **macOS (Homebrew)**: `brew install git-lfs`
   - **Debian/Ubuntu**: `sudo apt-get install git-lfs`
   - **Windows**: Download from [git-lfs.github.com](https://git-lfs.github.com/)

2. **Initialize Git LFS**:

   ```bash
   git lfs install
   ```

3. **Clone the Repository**:

   ```bash
   git clone https://github.com/farmrisk-in/farmrisk-models
   ```

   If you have already cloned the repository and see small pointer files in `data/`, you can pull the actual files by running:

   ```bash
   git lfs pull
   ```

### Data Directory Structure

The structure of the `data/` directory must be set up as follows:

```text
data/
├── grid_elevation.parquet
├── om_forecast_all.parquet
├── master_calibration_1D.csv
├── crop_calendar_parsed.csv
├── IMD_parquets/
│   ├── imd_tmax_daily.parquet
│   ├── imd_tmin_daily.parquet
│   └── imd_pcp_daily.parquet
└── training_data/
    ├── training_pcp_daily.parquet
    ├── training_tmax_daily.parquet
    └── training_tmin_daily.parquet
```

---

## Hugging Face Deployment & Syncing

Since Hugging Face Spaces have lower LFS limits and you expose this `data/` folder via a mounted storage bucket, we do **not** push the `data/` folder to the Hugging Face Space Git repository.

Instead, we use a GitHub Actions workflow that automatically syncs code modifications to the Hugging Face Space while excluding the `data/` folder entirely.

### Setup Steps for Hugging Face Sync:

1. **Configure GitHub Secrets**:
   Go to your GitHub Repository -> **Settings** -> **Secrets and variables** -> **Actions** and add a repository secret:
   - **Name**: `HF_TOKEN`
   - **Value**: Your Hugging Face Write Token (generated under Hugging Face **Settings** -> **Access Tokens**).

2. **How the Sync Workflow Works**:
   - The workflow ([.github/workflows/sync_to_hf.yml](file:///.github/workflows/sync_to_hf.yml)) runs on every push to the `main` branch.
   - It performs a standard Git checkout **without** fetching Git LFS files (saving runner bandwidth and time).
   - It uses `hf upload` with `--exclude` to push only the application code, Dockerfile, and metadata to your Hugging Face Space (`ShaanNeedsHugs/farmRisk`), ignoring the large datasets.
   
3. **Mounting the Bucket on Hugging Face**:
   Configure your Hugging Face Space (which uses the `Dockerfile`) to mount your external storage bucket (holding the data parquets) to the container path `/code/data/`. This way, the FastAPI app can read the datasets at startup and runtime seamlessly.

