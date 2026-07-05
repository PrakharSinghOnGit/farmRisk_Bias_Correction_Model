FROM python:3.12-slim

# Install system dependencies (libgomp1 is required for XGBoost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /code

# Create a non-root user (Hugging Face Spaces runs as user 1000)
RUN useradd -m -u 1000 user && \
    chown -R user:user /code

# Copy requirements and install them as the non-root user
COPY --chown=user:user requirements.txt /code/requirements.txt

USER user
ENV PATH="/home/user/.local/bin:${PATH}"

# Install Python packages
RUN pip install --no-cache-dir --user -r /code/requirements.txt

# Copy the application code, datasets, and cache folders
COPY --chown=user:user app /code/app
COPY --chown=user:user models /code/models

# Expose port 7860 (default port for Hugging Face Spaces)
EXPOSE 7860

# Command to launch the FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
