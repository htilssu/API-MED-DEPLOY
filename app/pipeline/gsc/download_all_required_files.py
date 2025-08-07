import os
import time
from pathlib import Path
from google.cloud import storage
from dotenv import load_dotenv
import google.generativeai as genai
from huggingface_hub import login
import logging
# ---------------------- CẤU HÌNH ----------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()
login(token=os.getenv("HUGGINGFACE_TOKEN"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ---------------------- CONSTANTS ----------------------
PROCESSED_DIR = "app/static/processed"
ANOMALY_MAP_DIR = "app/static/anomaly_maps"
ROI_OUTPUT_DIR = "app/static/roi_outputs"
GCS_BUCKET = "group_dataset-nt"
GCS_BUCKET_2 = "rag_3"
GCS_FOLDER = "handle_data"
LOCAL_SAVE_DIR = "app/static/"
LOCAL_SAVE_DIR_2 = "app/processed/"
GCS_KEY_PATH = storage.Client.from_service_account_json("app/iam-key.json")

REQUIRED_FILES = [
    "faiss_index.bin",
    "faiss_index_anomaly.bin",
    
    "faiss_index_bacterial_infections.bin",
    "faiss_index_fungal_infections.bin",
    "faiss_index_parasitic_infections.bin",
    "faiss_index_virus.bin",
    
    "faiss_index_anomaly_bacterial_infections.bin",
    "faiss_index_anomaly_fungal_infections.bin",
    "faiss_index_anomaly_parasitic_infections.bin",
    "faiss_index_anomaly_virus.bin",

    "labels.npy",
    "labels_anomaly.npy",

    "labels_bacterial_infections.npy",
    "labels_fungal_infections.npy",
    "labels_parasitic_infections.npy",
    "labels_virus.npy",

    "labels_anomaly_bacterial_infections.npy",
    "labels_anomaly_fungal_infections.npy",
    "labels_anomaly_parasitic_infections.npy",
    "labels_anomaly_virus.npy"
]

REQUIRED_FILES_2 = [
    "faiss_index.bin",
    "faiss_index_anomaly.bin",
    "labels.npy",
    "labels_anomaly.npy",

]

def get_gcs_client():
    try:
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        return storage.Client.from_service_account_json(credentials_path)
    except Exception as e:
        logging.error(f"Lỗi tạo Google Cloud Client: {e}")
        raise

def download_gcs_file(bucket_name: str, source_blob_name: str, destination_file_name: str, retries: int = 5):
    for attempt in range(1, retries + 1):
        try:
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(source_blob_name)
            blob.download_to_filename(destination_file_name, timeout=120)
            logging.info(f"Tải thành công {source_blob_name} → {destination_file_name}")
            return
        except Exception as e:
            logging.warning(f"Lỗi tải {source_blob_name} (lần {attempt}/{retries}): {e}")
            if attempt < retries:
                wait_time = 5 * attempt
                logging.info(f"Đợi {wait_time}s trước lần thử lại...")
                time.sleep(wait_time)
            else:
                logging.error(f"Thất bại sau {retries} lần: {source_blob_name}")

def download_all_required_files():
    Path(LOCAL_SAVE_DIR_2).mkdir(parents=True, exist_ok=True)
    for file in REQUIRED_FILES_2:
        gcs_path = f"{GCS_FOLDER}/{file}"
        local_path = os.path.join(LOCAL_SAVE_DIR_2, file)
        download_gcs_file(GCS_BUCKET_2, gcs_path, local_path)