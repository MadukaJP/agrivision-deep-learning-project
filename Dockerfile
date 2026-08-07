# AgriVision AI — Docker image for Render deployment
# Uses Python 3.12 so tensorflow==2.21.0 (cp312 Linux wheel) installs cleanly,
# avoiding Render's native-python runtime.txt issues entirely.

FROM python:3.12-slim-bookworm

# System libraries required by opencv-python-headless and TensorFlow at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better Docker layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project (model weights included via models/saved_models/).
COPY . .

# Render web services expect a process bound to 0.0.0.0:$PORT.
# Default to 8000 for local docker run.
EXPOSE 8000

# --chdir /app/app lets `from treatment_data import ...` resolve and app:app
# points at app/app.py. Single worker keeps memory low for TF's ~500MB load.
CMD ["sh", "-c", "gunicorn --chdir /app/app --bind 0.0.0.0:${PORT:-8000} --workers 1 --threads 4 app:app"]