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

# Cài lib runtime tối thiểu (không cần dev tools nữa)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies từ builder stage
COPY --from=builder /install /usr/local

# Copy code vào
COPY . .

# Env Cloud Run yêu cầu
ENV PORT=8080

# Chạy app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "3"]
