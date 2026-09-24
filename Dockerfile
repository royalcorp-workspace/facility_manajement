FROM python:3.11-slim

# Set environment variables untuk Python & Debian packaging
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Pasang paket sistem Debian minimal untuk OpenCV, rendering font, video streaming, dan healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Salin requirements.txt terlebih dahulu untuk memanfaatkan Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh kode proyek ke /app
COPY . .

# Port Web Streaming Hub (FastAPI / MJPEG)
EXPOSE 8070

# Jalankan Facility Management Master Orchestrator
CMD ["python", "main.py"]
