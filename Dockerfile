FROM python:3.12-slim

WORKDIR /code

RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*


    
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Sử dụng biến môi trường PORT để chạy đúng cổng Cloud Run yêu cầu
ENV PORT=8080

EXPOSE 8080


# Expose cổng 8080


# Chạy ứng dụng bằng uvicorn ở đúng cổng
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "3"]
