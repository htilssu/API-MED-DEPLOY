import cv2
import os
import numpy as np
import time
import torch
import faiss
import json
from pathlib import Path
from google.cloud import storage
from transformers import CLIPProcessor, CLIPModel
from dotenv import load_dotenv
import google.generativeai as genai
from transformers import pipeline
from huggingface_hub import login
import timm
from PIL import Image
from torchvision import transforms
import logging
# from transformers import Blip2Processor, Blip2ForConditionalGeneration
import textwrap
import re
from sklearn.metrics.pairwise import cosine_similarity
import requests
import xml.etree.ElementTree as ET
from typing import Optional, List, Dict, Tuple
from io import BytesIO
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from app.redis_client import redis_client, save_result_to_redis,get_result_by_key
import hashlib
from app.db.mongo import db
from geopy.geocoders import Nominatim
import time
import io

from app.pipeline.preprocess_image import preprocess_image
from app.pipeline.AI_Agent.generate_description import generate_description_with_Gemini
from app.pipeline.gsc.download_all_required_files import download_all_required_files
from app.pipeline.anomaly import generate_anomaly_overlay,save_anomaly_outputs,embed_anomaly_heatmap
from app.pipeline.embedding import embed_image_clip
from app.pipeline.AI_Agent.different_question import generate_discriminative_questions
from app.pipeline.AI_Agent.diagnose_group import generate_diagnosis_with_gemini
from app.pipeline.AI_Agent.final_diagnose import select_final_diagnosis_with_llm
user_collection = db["user"]



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
login(token=os.getenv("HUGGINGFACE_TOKEN"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


app = FastAPI()

load_dotenv()

# ---------------------- CONSTANTS ----------------------
ROI_OUTPUT_DIR = "app/static/roi_outputs"
GCS_BUCKET = "group_dataset-nt"
GCS_FOLDER = "handle_data"
LOCAL_SAVE_DIR = "app/static/"
GCS_DATASET = f"dataset"
GCS_DATASET_PATH = f"{GCS_DATASET}/dataset.json"
LOCAL_DATASET_PATH = "app/static/json/dataset.json"
GCS_BUCKET_2 = "rag_3"
LOCAL_SAVE_DIR_2 = "app/processed"




GCS_KEY_PATH = storage.Client.from_service_account_json("app/iam-key.json")



device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
vit_model = timm.create_model("vit_base_patch16_224", pretrained=True).to(device)
vit_model.eval()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def generate_disease_name(disease_name: str) -> str:
    prompt = f"""
    Bạn là một chuyên gia y tế, bạn có khả năng chuyển hóa tên bệnh về tên chuẩn nhất của nó (tên khoa học)
    Tên bệnh được truyền vào là: {disease_name}
    Hãy chuyển hóa tên bệnh đó về tên khoa học chuẩn nhất.
    Trả về tên bệnh đã chuyển hóa.
    Nếu không thể chuyển hóa, hãy trả về "Không thể chuyển hóa tên bệnh".
    """
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        result = response.text.strip()
        result = re.sub(r"^(?:\w+)?\n|\n$", "", result).strip()
        if not result:
            return "Không thể dịch tên bệnh"
        logger.info(f"Dịch tên bệnh: {disease_name} -> {result}")
        return result
    except Exception as e:
        logger.error(f"Lỗi khi dịch tên bệnh: {e}")
        return "Xảy ra lỗi trong quá trình dịch tên bệnh"





def search_faiss_index(embedding: np.ndarray, index_path: str, label_path: str, top_k: int = 5):
    index = faiss.read_index(index_path)
    labels = np.load(label_path, allow_pickle=True)
    distances, indices = index.search(embedding, top_k)
    top_labels = [labels[idx] for idx in indices[0]]
    return list(zip(top_labels, distances[0]))

def search_faiss_anomaly_index(embedding: np.ndarray, index_path: str, label_path: str, top_k: int = 5):
    return search_faiss_index(embedding, index_path, label_path, top_k)

def normalize_group_name(group_name: str) -> str:
    """
    Chuẩn hoá tên nhóm bệnh:
    - Chuyển thành chữ thường
    - Xoá khoảng trắng dư thừa
    - Thay khoảng trắng bằng dấu gạch dưới
    - Loại bỏ ký tự đặc biệt nếu cần (nếu tên nhóm có dấu chấm, dấu ngoặc,...)

    Ví dụ: 'Fungal infections' → 'fungal_infections'
    """
    group_name = group_name.lower()
    group_name = group_name.strip()
    group_name = re.sub(r"\s+", "_", group_name)         
    group_name = re.sub(r"[^a-z0-9_]", "", group_name)     
    return group_name

def aggregate_combined_results(combined_results):
    score_dict = {}
    for label, distance in combined_results:
        sim = 1 / (1 + distance)
        score_dict[label] = score_dict.get(label, 0) + sim

    total_score = sum(score_dict.values())
    normalized_scores = {label: (score / total_score) * 100 for label, score in score_dict.items()}
    sorted_scores = sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_scores


def format_diagnosis_output(diagnosis):
    if isinstance(diagnosis, tuple) and len(diagnosis) == 2:
        label, _ = diagnosis
        return str(label)
    elif isinstance(diagnosis, (str, np.str_)):
        return str(diagnosis)
    else:
        return str(diagnosis)

    



def detailed_group_analysis(image_vector: np.ndarray, anomaly_vector: np.ndarray, group_name: str, top_k: int = 5):
    print(f"\nSo sánh trong nhóm bệnh: {group_name}")
    index_path_full = os.path.join(LOCAL_SAVE_DIR, f"faiss_index_{group_name}.bin")
    label_path_full = os.path.join(LOCAL_SAVE_DIR, f"labels_{group_name}.npy")

    index_path_anomaly = os.path.join(LOCAL_SAVE_DIR, f"faiss_index_anomaly_{group_name}.bin")
    label_path_anomaly = os.path.join(LOCAL_SAVE_DIR, f"labels_anomaly_{group_name}.npy")

    print(f"\nKết quả so khớp ảnh thường với {group_name}:")
    full_group_results = search_faiss_index(
        embedding=image_vector,
        index_path=index_path_full,
        label_path=label_path_full,
        top_k=top_k
    )
    for label, score in full_group_results:
        similarity = 1 / (1 + score)
        print(f"  → {label} (similarity: {similarity*100:.2f}%)")

    print(f"\nKết quả so khớp anomaly heatmap với {group_name}:")
    anomaly_group_results = search_faiss_index(
        embedding=anomaly_vector,
        index_path=index_path_anomaly,
        label_path=label_path_anomaly,
        top_k=top_k
    )
    for label, score in anomaly_group_results:
        similarity = 1 / (1 + score)
        print(f"  → {label} (similarity: {similarity*100:.2f}%)")

    # ========== GỘP NHÃN VÀ VOTING ==========
    print(f"\nGộp nhãn từ 2 pipeline (ảnh thường + anomaly) trong nhóm '{group_name}':")

    combined_results = full_group_results + anomaly_group_results

    # Tính điểm similarity
    label_scores_raw = {}
    for label, distance in combined_results:
        similarity = 1 / (1 + distance)
        label_scores_raw[label] = label_scores_raw.get(label, 0) + similarity

    # Chuẩn hóa thành phần trăm
    total_similarity = sum(label_scores_raw.values())
    label_scores_percent = {label: (score / total_similarity) * 100 for label, score in label_scores_raw.items()}

    # Sắp xếp theo similarity %
    sorted_labels = sorted(label_scores_percent.items(), key=lambda x: x[1], reverse=True)

    print("Tổng điểm similarity trong nhóm bệnh (%):")
    for label, percent in sorted_labels:
        print(f"  → {label}: {percent:.2f}%")
    
    return sorted_labels

def normalize_diagnosis(diagnosis: str) -> str:
    """
    Chuẩn hóa diagnosis về định dạng chuẩn có dạng 'fungal_infections'.
    Chuyển các dấu gạch ngang, khoảng trắng thành gạch dưới.
    """
    return diagnosis.strip().lower().replace("-", "_").replace(" ", "_")




async def start_diagnois(image: UploadFile = File(...), user_id: Optional[str] = None):
    try:
        start_time = time.time()
        
        # --- B1: Kiểm tra input ---
        step_start = time.time()
        if not image:
            raise HTTPException(status_code=400, detail="Ảnh không được để trống")
        if not image.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            raise HTTPException(status_code=400, detail="Ảnh phải có định dạng .jpg, .jpeg hoặc .png")
        if not user_id:
            raise HTTPException(status_code=400, detail="ID người dùng không được để trống")
        print(f"[TIME] Kiểm tra input: {time.time() - step_start:.3f} giây")

        # --- B2: Đọc ảnh ---
        step_start = time.time()
        image_data = await image.read()
        print(f"[TIME] Đọc ảnh: {time.time() - step_start:.3f} giây")
        
        # --- B3: Sinh mô tả bằng Gemini ---
        step_start = time.time()
        description = generate_description_with_Gemini(image_data)
        print(f"[TIME] Sinh mô tả ảnh: {time.time() - step_start:.3f} giây")
        if description in ["Không phải ảnh da liễu", "Không phải ảnh da liễu."]:
            raise HTTPException(status_code=400, detail="Ảnh không phải là ảnh da liễu")

        # --- B4: Tiền xử lý ảnh ---
        step_start = time.time()
        preprocessed_pil, preprocessed_np = preprocess_image(image_data)
        print(f"[TIME] Tiền xử lý ảnh: {time.time() - step_start:.3f} giây")
        
        # --- B5: Tính embedding và tìm kiếm ban đầu ---
        step_start = time.time()
        test_vector = embed_image_clip(preprocessed_pil)
        test_results = search_faiss_index(
            embedding=test_vector,
            index_path=os.path.join(LOCAL_SAVE_DIR_2, "faiss_index.bin"),
            label_path=os.path.join(LOCAL_SAVE_DIR_2, "labels.npy"),
            top_k=15
        )
        test_results_anomaly = search_faiss_anomaly_index(
            embedding=test_vector,
            index_path=os.path.join(LOCAL_SAVE_DIR_2, "faiss_index_anomaly.bin"),
            label_path=os.path.join(LOCAL_SAVE_DIR_2, "labels_anomaly.npy"),
            top_k=15
        )
        print(f"[TIME] Tìm kiếm embedding ban đầu: {time.time() - step_start:.3f} giây")

        # --- B6: Kết hợp kết quả ban đầu ---
        step_start = time.time()
        test_combined_results = test_results + test_results_anomaly
        label_scores_test = {}
        for label, distance in test_combined_results:
            similarity = 1 / (1 + distance)
            label_scores_test[label] = label_scores_test.get(label, 0) + similarity
        total_similarity_test = sum(label_scores_test.values())
        label_scores_percent_test = {label: (score / total_similarity_test) * 100 for label, score in label_scores_test.items()}
        sorted_labels_test = sorted(label_scores_percent_test.items(), key=lambda x: x[1], reverse=True)
        print(f"[TIME] Xử lý kết quả ban đầu: {time.time() - step_start:.3f} giây")

        if sorted_labels_test[0][0] == "non-infectious diseases":
            raise HTTPException(status_code=400, detail="Không thể chẩn đoán bệnh này vì hiện tại bệnh thuộc vào nhóm bệnh chưa hỗ trợ")

        # --- B7: Tìm kiếm full image ---
        step_start = time.time()
        full_image_vector = embed_image_clip(preprocessed_pil)
        full_results = search_faiss_index(
            embedding=full_image_vector,
            index_path=os.path.join(LOCAL_SAVE_DIR, "faiss_index.bin"),
            label_path=os.path.join(LOCAL_SAVE_DIR, "labels.npy"),
            top_k=15
        )
        print(f"[TIME] Tìm kiếm full image: {time.time() - step_start:.3f} giây")

        # --- B8: Phân tích anomaly ---
        step_start = time.time()
        anomaly_overlay, anomaly_map = generate_anomaly_overlay(preprocessed_pil)
        overlay_path, anomaly_map_path = save_anomaly_outputs(anomaly_overlay, anomaly_map, image.filename)
        anomaly_vector = embed_anomaly_heatmap(overlay_path)
        anomaly_results = search_faiss_anomaly_index(
            embedding=anomaly_vector,
            index_path=os.path.join(LOCAL_SAVE_DIR, "faiss_index_anomaly.bin"),
            label_path=os.path.join(LOCAL_SAVE_DIR, "labels_anomaly.npy"),
            top_k=15
        )
        print(f"[TIME] Phân tích anomaly: {time.time() - step_start:.3f} giây")

        # --- B9: Kết hợp full image + anomaly ---
        step_start = time.time()
        combined_results = full_results + anomaly_results
        label_scores_raw = {}
        for label, distance in combined_results:
            similarity = 1 / (1 + distance)
            label_scores_raw[label] = label_scores_raw.get(label, 0) + similarity
        total_similarity = sum(label_scores_raw.values())
        label_scores_percent = {label: (score / total_similarity) * 100 for label, score in label_scores_raw.items()}
        sorted_labels = sorted(label_scores_percent.items(), key=lambda x: x[1], reverse=True)
        print(f"[TIME] Kết hợp full image + anomaly: {time.time() - step_start:.3f} giây")

        # --- B10: Chẩn đoán với Gemini ---
        step_start = time.time()
        group_name_raw = sorted_labels[0][0].split("/")[0]
        normalized_group_name = normalize_group_name(group_name_raw)
        diagnosis = generate_diagnosis_with_gemini(description, combined_results)
        print(f"[TIME] Chẩn đoán nhóm bệnh gemini: {time.time() - step_start:.3f} giây")
        
        step_start = time.time()
        normalized_group_diagnosis = normalize_diagnosis(diagnosis)
        print(f"[TIME] Chẩn đoán nhóm bệnh: {time.time() - step_start:.3f} giây")

        # --- B11: Phân tích nhóm bệnh chi tiết ---
        step_start = time.time()
        combined_results = detailed_group_analysis(
            image_vector=full_image_vector,
            anomaly_vector=anomaly_vector,
            group_name=normalized_group_diagnosis,
            top_k=15
        )
        print(f"[TIME] Phân tích nhóm bệnh chi tiết: {time.time() - step_start:.3f} giây")

        # --- B12: Lưu kết quả vào Redis ---
        step_start = time.time()
        disease_primary = [label for label, _ in combined_results]
        result_data = {
            "description": description,
            "disease_primary": disease_primary,
            "normalized_group_name": normalized_group_diagnosis,
        }
        saved = await save_result_to_redis(key=user_id, value=result_data)
        print(f"[TIME] Lưu kết quả vào Redis: {time.time() - step_start:.3f} giây")

        # --- Tổng thời gian ---
        print(f"[TIME] Tổng thời gian xử lý: {time.time() - start_time:.3f} giây")

        if not saved:
            raise HTTPException(status_code=500, detail="Lỗi khi lưu kết quả vào Redis")

        return JSONResponse(content=result_data, status_code=200)

    except HTTPException as e:
        logger.error(f"Lỗi HTTP: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Lỗi trong quá trình chẩn đoán: {e}")
        raise HTTPException(status_code=500, detail="Lỗi trong quá trình chẩn đoán") from e


async def get_diagnosis_result(key: str):
    result = await get_result_by_key(key)
    if not result:
        raise HTTPException(status_code=404, detail="Không tìm thấy kết quả")
    return result

async def get_discriminative_questions(key: str):
    try:
        start_time = time.time()
        if not key:
            raise HTTPException(status_code=400, detail="Key không được để trống")

        result = await get_result_by_key(key)
        if not result:
            raise HTTPException(status_code=404, detail="Không tìm thấy kết quả với key này")

        logger.info(f"Lấy kết quả chẩn đoán từ Redis với key: {key}")
        description = result.get("description", "")
        disease_primary = result.get("disease_primary", [])
        normalized_group_name = result.get("normalized_group_name", "")

        if not description or not disease_primary or not normalized_group_name:
            raise HTTPException(status_code=404, detail="Không đủ thông tin để tạo câu hỏi phân biệt")

        questions = generate_discriminative_questions(description, disease_primary, normalized_group_name)
        if not questions:
            raise HTTPException(status_code=500, detail="Không thể tạo câu hỏi phân biệt")

        logger.info(f"Câu hỏi phân biệt: {questions}")

        current_data_json = await get_diagnosis_result(key)
        if not current_data_json:
            raise HTTPException(status_code=404, detail="Không tìm thấy dữ liệu hiện tại")

        current_data = current_data_json
        current_data["questions"] = questions
        print(f"[TIME] Tạo câu hỏi phân biệt: {time.time() - start_time:.3f} giây")

        await redis_client.set(key, json.dumps(current_data))
        return JSONResponse(content={"questions": questions}, status_code=200)

    except HTTPException as e:
        logger.error(f"Lỗi HTTP: {e.detail}")
        raise e
    except Exception as e:
        logger.exception("Lỗi trong quá trình lấy câu hỏi phân biệt")
        raise HTTPException(status_code=500, detail="Lỗi trong quá trình lấy câu hỏi phân biệt") from e
    
async def submit_discriminative_questions(user_answers:str,key:str):
    try:
        if not key:
            raise HTTPException(status_code=400, detail="Key không được để trống")
        if not user_answers:
            raise HTTPException(status_code=400, detail="Câu trả lời không được để trống")
        current_data_json = await get_diagnosis_result(key)
        if not current_data_json:
            raise HTTPException(status_code=404, detail="Không tìm thấy dữ liệu hiện tại")
        current_data = current_data_json
        # Bắt đầu đưa ra chẩn đoán cuối cùng dựa trên câu trả lời người dùng
        description = current_data.get("description", "")
        disease_primary = current_data.get("disease_primary", [])
        questions = current_data.get("questions", [])
        normalized_group_name = current_data.get("normalized_group_name", "")
        if not description or not disease_primary or not questions:
            raise HTTPException(status_code=404, detail="Không đủ thông tin để đưa ra chẩn đoán cuối cùng")
        start_time = time.time()
        final_diagnosis = select_final_diagnosis_with_llm(
            caption=description,
            labels=disease_primary,
            questions=questions,
            answers=user_answers,
            group_disease_name= normalized_group_name
        )
        if not final_diagnosis:
            raise HTTPException(status_code=500, detail="Không thể đưa ra chẩn đoán cuối cùng")
        print(f"[TIME] Đưa ra chẩn đoán cuối cùng: {time.time() - start_time:.3f} giây")
        # Cập nhật kết quả vào Redis
        current_data["final_diagnosis"] = final_diagnosis
        await redis_client.set(key, json.dumps(current_data))
        return JSONResponse(content={"final_diagnosis": final_diagnosis}, status_code=200)
    except HTTPException as e:
        logger.error(f"Lỗi HTTP: {e.detail}")
        raise e

    
def translate_disease_name(disease_name: str) -> str:
    prompt = f"""
    Bạn là một chuyên gia y tế, bạn có khả năng dịch tên bệnh từ tiếng việt sang tiếng Anh (Tên khoa học của bệnh đó).
    Tên bệnh được truyền vào là: {disease_name}
    Hãy dịch tên bệnh đó sang tên khoa học .
    Trả về tên bệnh đã dịch.
    Chỉ trả về tên bệnh đã dịch, không thêm giải thích, không in Markdown, không thêm ký tự thừa.
    """
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        result = response.text.strip()
        result = re.sub(r"^(?:\w+)?\n|\n$", "", result).strip()
        if not result:
            return "Không thể dịch tên bệnh"
        logger.info(f"Dịch tên bệnh: {disease_name} -> {result}")
        return result
    except Exception as e:
        logger.error(f"Lỗi khi dịch tên bệnh: {e}")
        return "Xảy ra lỗi trong quá trình dịch tên bệnh"
    
def search_disease_in_json(file_path: str, disease_name: str) -> List[Dict]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            logger.warning("File JSON không phải là danh sách.")
            return []
        results = [
            entry for entry in data
            if isinstance(entry, dict) and disease_name.lower() in entry.get("Tên bệnh", "").lower()
        ]
        logger.info(f"Tìm thấy {len(results)} kết quả trong JSON cho bệnh: {disease_name}")
        return results
    except Exception as e:
        logger.error(f"Lỗi tìm kiếm trong JSON: {e}")
        return []
    
def search_medlineplus(ten_khoa_hoc: str) -> Optional[str]:
    if not ten_khoa_hoc or ten_khoa_hoc == "Không tìm thấy":
        logger.warning("Không có tên bệnh truyền vào cho MedlinePlus")
        return None
    logger.info(f"Tìm kiếm thông tin bệnh '{ten_khoa_hoc}' trên MedlinePlus...")
    url = 'https://wsearch.nlm.nih.gov/ws/query'
    params = {'db': 'healthTopics', 'term': ten_khoa_hoc}
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            cleaned_content = clean_xml_content(response.content)
            logger.info("Tìm kiếm MedlinePlus thành công")
            return cleaned_content
        logger.warning(f"Yêu cầu MedlinePlus thất bại, mã trạng thái: {response.status_code}")
        return None
    except Exception as e:
        logger.error(f"Lỗi tìm kiếm MedlinePlus: {e}")
        return None
def clean_text_json(data):
    if isinstance(data, dict):
        return {key: clean_text_json(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [clean_text_json(item) for item in data]
    else:
        return clean_text(str(data))

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return str(text)
    text = re.sub(r'[\\\n\r\t\*\]]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def clean_xml_content(xml_content: bytes) -> str:
    try:
        root = ET.fromstring(xml_content)
        text_nodes = [elem.text.strip() for elem in root.iter() if elem.text and elem.text.strip()]
        return ' '.join(text_nodes)
    except ET.ParseError as e:
        logger.error(f"Lỗi phân tích XML: {e}")
        return ""

def extract_medical_info(text: str) -> Dict:
    prompt = f"""
    Dịch văn bản về tiếng Việt
    Bạn là một chuyên gia y tế, bạn có khả năng trích xuất thông tin y khoa từ văn bản.
    Hãy trích xuất thông tin y khoa từ văn bản dưới dạng JSON hợp lệ **không chứa Markdown**.
    Chỉ lấy kết quả đầu tiên mà bạn tìm được
    Hãy trích xuất thật chi tiết
    Văn bản đầu vào là:
    {text}
    {{
        "Tên bệnh": "",
        "Tên khoa học": "",
        "Triệu chứng": "",
        "Vị trí xuất hiện": "",
        "Nguyên nhân": "",
        "Tiêu chí chẩn đoán": "",
        "Chẩn đoán phân biệt": "",
        "Điều trị": "",
        "Phòng bệnh": "",
        "Các loại thuốc": [{{"Tên thuốc": "", "Liều lượng": "", "Thời gian sử dụng": ""}}]
    }}
    - Nếu không có thông tin, đặt giá trị "Không tìm thấy".
    - Không thêm giải thích, không in Markdown, không thêm ký tự thừa.
    """
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        raw_text = response.text if hasattr(response, "text") else response.parts[0].text
        raw_text = re.sub(r"^```json\n|\n```$", "", raw_text)
        extracted_info = json.loads(raw_text)
        cleaned_info = clean_text_json(extracted_info)
        logger.info("Trích xuất thông tin y khoa từ MedlinePlus thành công")
        return cleaned_info
    except Exception as e:
        logger.error(f"Lỗi trích xuất thông tin y khoa: {e}")
        return {}
    
def download_from_gcs():
    storage_client = GCS_KEY_PATH
    bucket = storage_client.bucket(GCS_BUCKET)
    files_to_download = [
        (GCS_DATASET_PATH, LOCAL_DATASET_PATH),
    ]
    for gcs_path, local_path in files_to_download:
        blob = bucket.blob(gcs_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)
        logger.info(f"Tải về {gcs_path} to {local_path}")



async def knowledge(disease_name: str):
    try:
        # if not disease_name:
        #     raise HTTPException(status_code=400, detail="Cần cung cấp tên bệnh")
        # disease_name = generate_disease_name(disease_name)
        # if not disease_name or disease_name == "Không thể chuyển hóa tên bệnh":
        #     raise HTTPException(status_code=400, detail="Không thể chuyển hóa tên bệnh")
        # translated_name= translate_disease_name(disease_name)
        # search_result = search_disease_in_json(LOCAL_DATASET_PATH, translated_name)
        # if not search_result:
        start_time = time.time()
        medline_result = search_medlineplus(disease_name)
        if medline_result:
            extracted_info = extract_medical_info(medline_result)   
            if extracted_info:
                search_result = [extracted_info]
        if not search_result:
            raise HTTPException(status_code=404, detail=f"Không tìm thấy thông tin cho bệnh: {disease_name}")
        logger.info(f"Tra cứu thông tin bệnh: {disease_name}")
        print(f"[TIME] Tra cứu thông tin bệnh: {time.time() - start_time:.3f} giây")
        return JSONResponse(content={"disease_info": search_result}, status_code=200)
    except Exception as e:
        logger.error(f"Lỗi khi tra cứu thông tin bệnh: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi tra cứu thông tin bệnh: {str(e)}")
    
async def knowledge_translate(disease_name: str):
    try:
        if not disease_name:
            raise HTTPException(status_code=400, detail="Cần cung cấp tên bệnh")
        translated_name = translate_disease_name(disease_name)
        if not translated_name or translated_name == "Không thể dịch tên bệnh":
            raise HTTPException(status_code=400, detail="Không thể dịch tên bệnh")
        logger.info(f"Dịch tên bệnh: {disease_name} -> {translated_name}")
        print(f"Dịch tên bệnh: {disease_name} -> {translated_name}")
        print("Tra cứu thông tin bệnh...")
        medline_result = search_medlineplus(translated_name)
        if medline_result:
            extracted_info = extract_medical_info(medline_result)   
            if extracted_info:
                search_result = [extracted_info]
        if not search_result:
            raise HTTPException(status_code=404, detail=f"Không tìm thấy thông tin cho bệnh: {translated_name}")
        logger.info(f"Tra cứu thông tin bệnh: {translated_name}")
        return JSONResponse(content={"translated_name": translated_name, "disease_info": search_result}, status_code=200)
    except HTTPException as e:
        logger.error(f"Lỗi HTTP: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Lỗi khi dịch tên bệnh: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi dịch tên bệnh: {str(e)}")