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

# Cài lib runtime tối thiểu và tools cần thiết cho download script
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

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
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}

