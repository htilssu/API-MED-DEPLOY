# =========================
# Stage 1: Build dependencies
# =========================
FROM python:3.12-slim AS builder

WORKDIR /app

# Cài dependency cần để build (numpy, scipy, torch,... hay cần compile)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-dev \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements và install vào 1 folder riêng
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# =========================
# Stage 2: Runtime image
# =========================
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    wget \
    curl \
    apt-transport-https \
    ca-certificates \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && apt-get update -y && apt-get install google-cloud-cli -y
RUN gsutil init
RUN gsutil -m cp -r gs://demer/i2/* ./index2

# Copy dependencies từ builder stage
COPY --from=builder /install /usr/local

# Copy code vào
COPY . .

# Make download script executable
RUN chmod +x download_index_files.sh

# Run download script to get index files
RUN ./download_index_files.sh || echo "Warning: Could not download index files during build. Run ./download_index_files.sh manually if needed."

# Env Cloud Run yêu cầu
ENV PORT=8000

# Chạy app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

