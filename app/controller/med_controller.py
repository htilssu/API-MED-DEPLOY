import cv2
import os
import numpy as np
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
from typing import Optional, List, Dict
from io import BytesIO
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from app.redis_client import redis_client, save_result_to_redis,get_result_by_key
import hashlib
from app.db.mongo import db

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

load_dotenv()
login(token=os.getenv("HUGGINGFACE_TOKEN"))

GCS_BUCKET = "kltn-2025"
GCS_IMAGE_PATH = "uploaded_images/"
GCS_KEY_PATH = storage.Client.from_service_account_json("app/iamkey.json")
VECTOR_FILE = "static/processed/embedded_vectors.json"
GCS_FOLDER = "handle_data"
GCS_DATASET = f"dataset"
GCS_DATASET_PATH = f"{GCS_DATASET}/dataset.json"
GCS_INDEX_PATH = f"{GCS_FOLDER}/faiss_index.bin"
GCS_LABELS_PATH = f"{GCS_FOLDER}/labels.npy"
GCS_TEXT_INDEX_PATH = f"{GCS_FOLDER}/faiss_text_index.bin"
GCS_TEXT_LABELS_PATH = f"{GCS_FOLDER}/text_labels.npy"
GCS_ANOMALY_INDEX_PATH = f"{GCS_FOLDER}/faiss_index_anomaly.bin"
GCS_ANOMALY_LABELS_PATH = f"{GCS_FOLDER}/labels_anomaly.npy"
LOCAL_INDEX_PATH = "app/static/faiss/faiss_index.bin"
LOCAL_LABELS_PATH = "app/static/labels/labels.npy"
LOCAL_TEXT_INDEX_PATH = "app/static/faiss/faiss_text_index.bin"
LOCAL_TEXT_LABELS_PATH = "app/static/labels/text_labels.npy"
LOCAL_ANOMALY_INDEX_PATH = "app/static/faiss/faiss_index_anomaly.bin"
LOCAL_ANOMALY_LABELS_PATH = "app/static/labels/labels_anomaly.npy"
LOCAL_DATASET_PATH = "app/static/json/dataset.json"
INDEX_DIM = 512

index = None
labels = []
anomaly_index = None
anomaly_labels = []

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
vit_model = timm.create_model("vit_base_patch16_224", pretrained=True).to(device)
vit_model.eval()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

users_collection = db["users"]

def download_from_gcs():
    storage_client = GCS_KEY_PATH
    bucket = storage_client.bucket(GCS_BUCKET)
    files_to_download = [
        (GCS_INDEX_PATH, LOCAL_INDEX_PATH),
        (GCS_LABELS_PATH, LOCAL_LABELS_PATH),
        (GCS_TEXT_INDEX_PATH, LOCAL_TEXT_INDEX_PATH),
        (GCS_TEXT_LABELS_PATH, LOCAL_TEXT_LABELS_PATH),
        (GCS_ANOMALY_INDEX_PATH, LOCAL_ANOMALY_INDEX_PATH),
        (GCS_ANOMALY_LABELS_PATH, LOCAL_ANOMALY_LABELS_PATH),
        (GCS_DATASET_PATH, LOCAL_DATASET_PATH),
    ]
    for gcs_path, local_path in files_to_download:
        blob = bucket.blob(gcs_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)
        logger.info(f"Tải về {gcs_path} to {local_path}")

def upload_to_gcs(local_path: str, destination_blob_name: str):
    client = storage.Client.from_service_account_json("app/iamkey.json")
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_path)
    logger.info(f"Đã upload {local_path} lên GCS tại: gs://{GCS_BUCKET}/{destination_blob_name}")

def preprocess_image(image_data: bytes) -> Optional[np.ndarray]:
    try:
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        equalized = cv2.equalizeHist(blurred)
        edges = cv2.Canny(equalized, 50, 150)
        return edges
    except Exception as e:
        logger.error(f"Lỗi xử lý ảnh: {e}")
        return None

def embed_image(image_data: bytes):
    try:
        image = Image.open(BytesIO(image_data)).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            embedding = model.get_image_features(**inputs)
        return embedding.cpu().numpy().astype(np.float32)
    except Exception as e:
        logger.error(f"Lỗi nhúng ảnh: {e}")
        return None

def generate_anomaly_map(image_data: bytes) -> Optional[np.ndarray]:
    try:
        img = Image.open(BytesIO(image_data)).convert("RGB")
        original_size = img.size
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            features = vit_model.forward_features(img_tensor)
        feature_map = features.mean(dim=1).squeeze().cpu().numpy()
        anomaly_map = (feature_map - np.min(feature_map)) / (np.ptp(feature_map) + 1e-6)
        anomaly_map = (anomaly_map * 255).astype(np.uint8)
        anomaly_map_resized = cv2.resize(anomaly_map, original_size, interpolation=cv2.INTER_CUBIC)
        return anomaly_map_resized
    except Exception as e:
        logger.error(f"Lỗi tạo Anomaly Map: {e}")
        return None

def embed_anomaly_map(anomaly_map: np.ndarray):
    try:
        anomaly_map_rgb = cv2.cvtColor(anomaly_map, cv2.COLOR_GRAY2RGB)
        image = Image.fromarray(anomaly_map_rgb)
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            embedding = model.get_image_features(**inputs)
        return embedding.cpu().numpy().astype(np.float32)
    except Exception as e:
        logger.error(f"Lỗi nhúng Anomaly Map: {e}")
        return None

def load_faiss_index():
    global index, labels, anomaly_index, anomaly_labels
    try:
        if os.path.exists(LOCAL_INDEX_PATH):
            index = faiss.read_index(LOCAL_INDEX_PATH)
            logger.info(f"FAISS Index tải thành công! Tổng số vector: {index.ntotal}")
        else:
            logger.warning("FAISS Index không tồn tại!")
        if os.path.exists(LOCAL_TEXT_LABELS_PATH):
            labels = np.load(LOCAL_TEXT_LABELS_PATH, allow_pickle=True).tolist()
            logger.info(f"Đã tải {len(labels)} nhãn bệnh từ labels.npy")
        else:
            logger.warning("labels.npy không tồn tại!")
        if os.path.exists(LOCAL_ANOMALY_INDEX_PATH):
            anomaly_index = faiss.read_index(LOCAL_ANOMALY_INDEX_PATH)
            logger.info(f"FAISS Anomaly Index tải thành công! Tổng số vector: {anomaly_index.ntotal}")
        else:
            logger.warning("FAISS Anomaly Index không tồn tại!")
        if os.path.exists(LOCAL_ANOMALY_LABELS_PATH):
            anomaly_labels = np.load(LOCAL_ANOMALY_LABELS_PATH, allow_pickle=True).tolist()
            logger.info(f"Đã tải {len(anomaly_labels)} nhãn bệnh từ labels_anomaly.npy")
        else:
            logger.warning("labels_anomaly.npy không tồn tại!")
    except Exception as e:
        logger.error(f"Lỗi tải FAISS Index: {e}")

def search_similar_images(query_vector, top_k=5):
    if index is None or index.ntotal == 0:
        logger.warning("FAISS index trống!")
        return []
    try:
        if query_vector.ndim == 1:
            query_vector = np.expand_dims(query_vector, axis=0)
        faiss.normalize_L2(query_vector)

        distances, indices = index.search(query_vector, top_k)
        logger.info(f"Chỉ số tìm thấy: {indices}")
        logger.info(f"Cosine similarities: {distances}")

        similar_results = []
        for idx, sim in zip(indices[0], distances[0]):
            if sim < 80:  # Loại bỏ nhãn có similarity dưới 0.8
                continue
            if 0 <= idx < len(labels):
                label_filename = list(labels.keys())[idx]
                label = labels[label_filename]
            else:
                logger.warning(f"Index {idx} vượt phạm vi labels ({len(labels)})!")
                label = "unknown"
            similar_results.append({
                "label": label,
                "cosine_similarity": float(sim)
            })
        return similar_results
    except Exception as e:
        logger.error(f"Lỗi tìm kiếm ảnh tương tự: {e}")
        return []

def search_anomaly_images(query_vector, top_k=5):
    if anomaly_index is None or anomaly_index.ntotal == 0:
        logger.warning("FAISS Anomaly Index trống!")
        return []
    try:
        if query_vector.ndim == 1:
            query_vector = np.expand_dims(query_vector, axis=0)
        faiss.normalize_L2(query_vector)

        distances, indices = anomaly_index.search(query_vector, top_k)
        logger.info(f"Chỉ số tìm thấy: {indices}")
        logger.info(f"Cosine similarities (anomaly): {distances}")

        similar_results = []
        for idx, sim in zip(indices[0], distances[0]):
            if sim < 80:  # Loại bỏ nhãn có similarity dưới 0.8
                continue    
            if 0 <= idx < len(anomaly_labels):
                label_filename = list(anomaly_labels.keys())[idx]
                label = anomaly_labels[label_filename]
            else:
                logger.warning(f"Index {idx} vượt phạm vi labels_anomaly ({len(anomaly_labels)})!")
                label = "unknown"
            similar_results.append({
                "label": label,
                "cosine_similarity": float(sim)
            })
        return similar_results
    except Exception as e:
        logger.error(f"Lỗi tìm kiếm ảnh anomaly: {e}")
        return []

def combine_labels(detailed_labels_normal: List[Dict], detailed_labels_anomaly: List[Dict]) -> str:
    """
    Kết hợp nhãn từ detailed_labels_normal và detailed_labels_anomaly, loại bỏ trùng lặp và nhãn có similarity < 0.8.
    Args:
        detailed_labels_normal: Danh sách dict chứa label và cosine_similarity từ ảnh gốc.
        detailed_labels_anomaly: Danh sách dict chứa label và cosine_similarity từ anomaly map.
    Returns:
        Chuỗi các nhãn được kết hợp, sắp xếp theo cosine_similarity giảm dần.
    """
    # Kết hợp tất cả nhãn từ cả hai nguồn
    all_labels = detailed_labels_normal + detailed_labels_anomaly
    
    # Chuẩn hóa và lọc nhãn
    filtered_labels = []
    seen_labels = {}  # Lưu nhãn và similarity cao nhất
    for item in all_labels:
        label = item["label"]
        sim = item["cosine_similarity"]
        # Chuẩn hóa similarity nếu ở thang 0-100
        normalized_sim = sim / 100.0 if sim > 1.0 else sim
        if normalized_sim < 0.8:  # Loại bỏ nhãn có similarity < 0.8
            continue
        # Giữ nhãn có similarity cao nhất nếu trùng lặp
        if label not in seen_labels or normalized_sim > seen_labels[label]:
            seen_labels[label] = normalized_sim
    
    # Tạo danh sách nhãn đã lọc và sắp xếp theo similarity giảm dần
    filtered_labels = [
        {"label": label, "cosine_similarity": sim}
        for label, sim in seen_labels.items()
    ]
    filtered_labels.sort(key=lambda x: x["cosine_similarity"], reverse=True)
    
    # Tạo chuỗi nhãn
    final_labels = " ".join(item["label"] for item in filtered_labels).strip()
    
    logger.info(f"Nhãn tổng hợp sau lọc và sắp xếp: {final_labels}")
    return final_labels

def generate_description_with_Gemini(image_data: bytes):
    try:
        img = Image.open(BytesIO(image_data))
        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = """
        Mô tả bức ảnh này bằng tiếng Việt, đây là ảnh y khoa nên hãy mô tả thật kỹ.
        Chỉ tập trung vào mô tả lâm sàng, không đưa ra chẩn đoán hay kết luận.
        Hãy mô tả các đặc điểm sau:
        - Vị trí của tổn thương (ví dụ: lòng bàn tay, mu bàn tay, ngón chân...)
        - Kích thước tổn thương (ước lượng theo mm hoặc cm)
        - Màu sắc (đồng nhất hay nhiều màu, đỏ, tím, hồng, v.v.)
        - Kết cấu bề mặt da (mịn, sần sùi, có vảy, loét...)
        - Độ rõ nét của các cạnh tổn thương (rõ ràng hay mờ, lan tỏa)
        - Tính đối xứng (tổn thương có đối xứng 2 bên hay không)
        - Phân bố (rải rác, tập trung thành đám, theo đường…)
        - Các đặc điểm bất thường khác nếu có (chảy máu, vảy, mụn nước, sưng nề…)
        Chỉ mô tả những gì có thể nhìn thấy trong ảnh, không đưa ra giả định hay chẩn đoán y khoa.
        Xóa mark down và các ký tự đặc biệt trong kết quả.
        """
        response = model.generate_content([prompt, img])
        caption = response.text.replace("\n", " ")
        logger.info("Tạo mô tả ảnh thành công với Gemini")
        return caption
    except Exception as e:
        logger.error(f"Lỗi khi tạo caption với Gemini: {e}")
        return None

def generate_medical_entities(image_caption, user_description):
    combined_description = f"1. Mô tả từ người dùng: {user_description}. 2. Mô tả từ ảnh: {image_caption}."
    print(combined_description)

    prompt = textwrap.dedent(f"""
        Tôi có 2 đoạn mô tả sau về một vùng da bị bất thường: {combined_description}
        Hãy chuẩn hóa cả hai mô tả, loại bỏ từ dư thừa, hợp nhất lại, và trích xuất các đặc trưng y khoa quan trọng.
        Mỗi đặc trưng nên được gắn nhãn thuộc một trong ba loại sau:
        - "Triệu chứng": mô tả biểu hiện, dấu hiệu lâm sàng (ví dụ: phát ban, ngứa, đỏ, bong tróc…)
        - "Vị trí xuất hiện": vùng cơ thể bị ảnh hưởng (ví dụ: mu bàn tay, cẳng chân, ngón tay…)
        - "Nguyên nhân": yếu tố gây ra tình trạng đó nếu có xuất hiện trong mô tả (ví dụ: côn trùng cắn, dị ứng, tiếp xúc hóa chất…)
        Trả về kết quả dạng JSON Array. Mỗi phần tử là một object gồm:
        - "entity": cụm từ y khoa
        - "type": "Triệu chứng", "Vị trí xuất hiện", hoặc "Nguyên nhân"
        Ví dụ đầu ra:
        [
          {{ "entity": "vết đỏ", "type": "Triệu chứng" }},
          {{ "entity": "cẳng chân", "type": "Vị trí xuất hiện" }},
          {{ "entity": "dị ứng thời tiết", "type": "Nguyên nhân" }}
        ]
        Chỉ liệt kê các đặc trưng có trong mô tả. Không suy luận thêm.
    """)
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content([prompt])
        text = response.text.strip()
        clean_text = re.sub(r"```(?:json)?|```", "", text).strip()
        result = json.loads(clean_text)
        decoded_result = json.dumps(result, ensure_ascii=False)
        return decoded_result 

    except Exception as e:
        print(f"Lỗi khi tạo mô tả với Gemini: {e}")
        return None

def compare_descriptions_and_labels(description: str, labels: str):
    prompt = textwrap.dedent(f"""
        Mô tả: "{description}"
        Nhãn: "{labels}"
        So sánh sự khác biệt giữa mô tả và nhãn bệnh. Sau đó, tạo ra 3 câu hỏi giúp phân biệt chính xác hơn.
        Trả về kết quả theo định dạng:
        1. ...
        2. ...
        3. ...
    """)
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content([prompt])
        text = response.text.strip()
        questions = re.findall(r"\d+\.\s+(.*)", text)
        logger.info("Tạo câu hỏi phân biệt thành công")
        return questions
    except Exception as e:
        logger.error(f"Lỗi khi gọi Gemini: {e}")
        return []

def filter_incorrect_labels_by_user_description(description: str, labels: list[str]) -> str:
    prompt = textwrap.dedent(f"""
        Mô tả bệnh của người dùng: "{description}"
        Danh sách các nhãn bệnh nghi ngờ: [{labels}]

        Nhiệm vụ:
        1. Phân tích mô tả và so sánh với từng nhãn bệnh.
        2. Loại bỏ các nhãn bệnh không phù hợp với mô tả. Giải thích lý do loại bỏ rõ ràng.
        3. Giữ lại các nhãn phù hợp nhất, sắp xếp theo mức độ phù hợp giảm dần.
        4. Trả thêm similarity của từng nhãn bệnh 
        Kết quả đầu ra phải ở định dạng JSON:
        {{
            "loai_bo": [{{"label": "nhãn không phù hợp", "ly_do": "...","similarity":""}}],
            "giu_lai": [{{"label": "nhãn phù hợp", "do_phu_hop": "cao/trung bình/thấp", "similarity":""}}]
        }}
    """)

    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content([prompt])
        text = response.text.strip()
        clean_text = re.sub(r"```(?:json)?|```", "", text).strip()
        result = json.loads(clean_text)
        return result 

    except Exception as e:
        print(f"Lỗi khi tạo mô tả với Gemini: {e}")
        return None

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
    
def append_disease_to_json(file_path: str, new_disease: dict):
    with open(file_path, 'r+', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            data = []
        data.append(new_disease)
        f.seek(0)
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.truncate()
    print(f"Added new disease: {new_disease.get('name', '')}")

def upload_json_to_gcs(bucket_name: str, destination_blob_name: str, source_file_name: str):
    client = storage.Client.from_service_account_json("app/iamkey            .json")
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"Uploaded {source_file_name} to {destination_blob_name}.")
    

def search_final(name):
    translate_name=translate_disease_name(name)
    print(f"Tên bệnh đã dịch: {translate_name}")
    search_json_result = search_disease_in_json(LOCAL_DATASET_PATH, translate_name)
    if search_json_result:
        print(f"Kết quả tìm kiếm trong file JSON: {search_json_result}")
    else:
        print(f"Không tìm thấy tên bệnh '{translate_name}' trong file JSON.")
        print ("Bắt đầu tìm kiếm bằng MedlinePlus...")
        search_medline_result = search_medlineplus(name)
        print(f"Kết quả tìm kiếm MedlinePlus: {search_medline_result}")
        print ("Bắt đầu trích xuất thông tin y khoa từ MedlinePlus...")
        extract_medical_info_result = extract_medical_info(search_medline_result)
        if extract_medical_info_result:
            print(f"Kết quả trích xuất thông tin y khoa: {extract_medical_info_result}")
            print("Đang thêm thông tin vào file JSON...")
            append_disease_to_json(LOCAL_DATASET_PATH, extract_medical_info_result)
            print("Upload file JSON lên GCS...")
            upload_json_to_gcs(GCS_BUCKET, GCS_DATASET_PATH, LOCAL_DATASET_PATH)

def translate_disease_name(disease_name: str) -> str:
    prompt = f"""
    Bạn là một chuyên gia y tế, bạn có khả năng dịch tên bệnh từ tiếng Anh sang tiếng Việt.
    Tên bệnh được truyền vào là: {disease_name}
    Hãy dịch tên bệnh đó sang tiếng Việt.
    Trả về tên bệnh đã dịch.
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

def clean_xml_content(xml_content: bytes) -> str:
    try:
        root = ET.fromstring(xml_content)
        text_nodes = [elem.text.strip() for elem in root.iter() if elem.text and elem.text.strip()]
        return ' '.join(text_nodes)
    except ET.ParseError as e:
        logger.error(f"Lỗi phân tích XML: {e}")
        return ""
    
def combine_user_questions_and_answers(user_questions, user_answers):
    if not user_questions or not user_answers:
        logger.warning("Không có câu hỏi hoặc câu trả lời từ người dùng")
        return []
    prompt = f"""Hãy tổng hợp các câu hỏi và câu trả lời của người dùng theo thứ tự 
    Danh sách câu hỏi{ user_questions}
    Danh sách câu trả lời{ user_answers}
    Ví dụ:Bạn có nổi mẫn đó không? Có.
    Trả về dạng chuỗi câu hỏi và câu trả lời, mỗi câu hỏi và câu trả lời cách nhau bằng dấu phẩy.
    """
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        result = response.text.strip()
        if not result:
            logger.warning("Không có kết quả trả về từ Gemini")
            return []
        logger.info("Tổng hợp câu hỏi và câu trả lời thành công")
        return result.split(", ")
    except Exception as e:
        logger.error(f"Lỗi khi tổng hợp câu hỏi và câu trả lời: {e}")
        return []

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

def process_image(image_data: bytes):
    processed = preprocess_image(image_data)
    if processed is None:
        logger.error("Lỗi xử lý ảnh, dừng quy trình.")
        return None, [], [], [], []

    processed_dir = Path("app/static/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_path = processed_dir / "processed_temp.jpg"
    cv2.imwrite(str(processed_path), processed)
    upload_to_gcs(str(processed_path), GCS_IMAGE_PATH + str(processed_path.name))

    embedding = embed_image(image_data)
    result_labels_simple = []
    detailed_labels_normal = []
    if embedding is not None:
        detailed_labels_normal = search_similar_images(embedding)
        result_labels_simple = [item["label"] for item in detailed_labels_normal]
        logger.info("🔍 Ảnh gốc:")
        for item in detailed_labels_normal:
            logger.info(f"- {item['label']} (cosine: {item['cosine_similarity']:.4f})")

    anomaly_map = generate_anomaly_map(image_data)
    anomaly_result_labels_simple = []
    detailed_labels_anomaly = []
    if anomaly_map is not None:
        anomaly_map_path = processed_dir / "anomaly_map_temp.jpg"
        cv2.imwrite(str(anomaly_map_path), anomaly_map)
        upload_to_gcs(str(anomaly_map_path), GCS_IMAGE_PATH + str(anomaly_map_path.name))

        anomaly_map_embedding = embed_anomaly_map(anomaly_map)
        if anomaly_map_embedding is not None:
            detailed_labels_anomaly = search_anomaly_images(anomaly_map_embedding)
            anomaly_result_labels_simple = [item["label"] for item in detailed_labels_anomaly]
            logger.info("Anomaly Map:")
            for item in detailed_labels_anomaly:
                logger.info(f"- {item['label']} (cosine: {item['cosine_similarity']:.4f})")
    final_labels = combine_labels(detailed_labels_anomaly, detailed_labels_normal)
    logger.info(f"Nhãn tổng hợp cuối cùng: {final_labels}")

    return final_labels, result_labels_simple, anomaly_result_labels_simple, detailed_labels_normal, detailed_labels_anomaly
async def start_diagnois(image: UploadFile = File(...),user_id: Optional[str] = None):
    try:
        download_from_gcs()
        load_faiss_index()
        if not image.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            raise HTTPException(status_code=400, detail="Ảnh phải có định dạng .jpg, .jpeg hoặc .png")
        image_data = await image.read()

        Key =user_id
        print(f"Key người dùng: {Key}")
        final_labels, result_labels_simple, anomaly_result_labels_simple, detailed_labels_normal, detailed_labels_anomaly = process_image(image_data)
        if not final_labels:
            raise HTTPException(status_code=500, detail="Không thể xử lý ảnh")

        image_description = generate_description_with_Gemini(image_data)
        if not image_description:
            raise HTTPException(status_code=500, detail="Không thể tạo mô tả ảnh")

        logger.info(f"Mô tả ảnh: {image_description}")

        result_data = {
            "final_labels": final_labels,
            "image_description": image_description,
            "result_labels": result_labels_simple,
            "anomaly_result_labels": anomaly_result_labels_simple,
            "detailed_labels_normal": detailed_labels_normal,
            "detailed_labels_anomaly": detailed_labels_anomaly
        }
        saved = await save_result_to_redis(key=Key, value=result_data)
        if not saved:
            logger.warning("Không thể lưu kết quả vào Redis")
            raise HTTPException(status_code=500, detail="Không thể lưu kết quả vào Redis")

        return JSONResponse(content=result_data, status_code=200)

    except HTTPException as e:
        logger.error(f"Lỗi HTTP: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Lỗi khác: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi trong quá trình chẩn đoán: {str(e)}")
    
async def get_diagnosis_result(key: str):
    result = await get_result_by_key(key)
    if not result:
        raise HTTPException(status_code=404, detail="Không tìm thấy kết quả")
    return result

async def submit_user_description(user_description: str, key: str):
    try:
        if not user_description or not key:
            raise HTTPException(status_code=400, detail="Thiếu mô tả ảnh hoặc key")
        result = await get_result_by_key(key)
        if not result:
            raise HTTPException(status_code=404, detail="Không tìm thấy kết quả chẩn đoán")
        current_data_json = await get_diagnosis_result(key)
        if not current_data_json:
            raise HTTPException(status_code=404, detail="Không tìm thấy dữ liệu chẩn đoán hiện tại")
        current_data = current_data_json
        current_data["user_description"] = user_description

        await redis_client.set(key, json.dumps(current_data), ex=3600)  
        return JSONResponse(content={"message": "Mô tả người dùng đã được lưu thành công"}, status_code=200)
    except HTTPException as e:
        logger.error(f"Lỗi khi lưu mô tả người dùng: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi lưu mô tả người dùng: {str(e)}")     
        
async def get_differentiation_questions(key:str):
    try:
        result = await get_result_by_key(key)
        if not result:
            raise HTTPException(status_code=404, detail="Không tìm thấy kết quả chẩn đoán")
        user_description = result.get("user_description", "")
        if not user_description or not key:
            raise HTTPException(status_code=400, detail="Thiếu mô tả ảnh hoặc key")
        result_medical_entities = generate_medical_entities(
            user_description or "Không có mô tả từ người dùng",
            result.get("image_description", "") or "Không có mô tả từ ảnh"
        )
        if not result_medical_entities:
            raise HTTPException(status_code=500, detail="Không thể tạo mô tả y khoa")
        
        questions = compare_descriptions_and_labels(result_medical_entities,result.get("final_labels", ""))
        if not questions:
            raise HTTPException(status_code=500, detail="Không thể tạo câu hỏi phân biệt")
        logger.info(f"Câu hỏi phân biệt: {questions}")
        current_data_json = await get_diagnosis_result(key)
        if not current_data_json:
            raise HTTPException(status_code=404, detail="Không tìm thấy dữ liệu chẩn đoán hiện tại")
        current_data= current_data_json
        current_data["questions"] = questions
        current_data["medical_entities"] = result_medical_entities

        await redis_client.set(key, json.dumps(current_data), ex=3600)  
        return JSONResponse(content=[{"questions": questions},{"medical_entites":result_medical_entities}], status_code=200)
    except HTTPException as e:
        logger.error(f"Lỗi khi tạo câu hỏi phân biệt: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi tạo câu hỏi phân biệt: {str(e)}")

async def submit_differation_questions(user_answers:dict,key:str):
    try:
        result = await get_result_by_key(key)
        if not result:
            raise HTTPException(status_code=404, detail="Không tìm thấy kết quả chẩn đoán")
        medical_entities = result.get("medical_entities", "")
        if not medical_entities:
            raise HTTPException(status_code=500, detail="Không có mô tả y khoa")
        if not user_answers or not isinstance(user_answers, dict):
            raise HTTPException(status_code=400, detail="Dữ liệu câu trả lời không hợp lệ")
        questions = result.get("questions", [])
        if not questions:
            raise HTTPException(status_code=404, detail="Không tìm thấy câu hỏi phân biệt")
        combined_answer = combine_user_questions_and_answers(questions, user_answers)

        combined_description= f"{medical_entities}\n\n{combined_answer}"
        final_labels = result.get("final_labels", "")
        print("\n--- Đang loại trừ nhãn không phù hợp ---")
        result_filter =filter_incorrect_labels_by_user_description(combined_description, final_labels)
        if not result_filter:
            print("Không có kết quả từ Gemini.")
            return
        refined_labels = result_filter.get("giu_lai", [])
        if not refined_labels:
            print("Không còn nhãn nào phù hợp. Đề xuất tham khảo bác sĩ.")
        else:
            print("Các nhãn còn lại sau loại trừ:")
        
        result_redis = []
        for label_info in refined_labels:
            label = label_info.get("label")
            ket_qua = "-".join(label.split("-")[1:])
            suitability = label_info.get("do_phu_hop")
            similarity= label_info.get("similarity", "")
            print(f"- {ket_qua} (Mức độ phù hợp: {suitability}) (Similarity: {similarity})")
            result_redis.append({"ketqua": ket_qua,"do_phu_hop": suitability, "similarity": similarity})
        current_data_json = await get_diagnosis_result(key)
        current_data_json["result"] = result_redis
        print(result_redis)
        await redis_client.set(key, json.dumps(current_data_json), ex=3600) 
        if not current_data_json:
            raise HTTPException(status_code=404, detail="Không tìm thấy dữ liệu chẩn đoán hiện tại")
        return JSONResponse(content=[{"result":result_redis}], status_code=200)
    except HTTPException as e:
        logger.error(f"Lỗi khi loại trừ nhãn không phù hợp: {e.detail}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi loại trừ nhãn không phù hợp: {str(e)}")
        
async def knowledge(disease_name: str):
    try:
        if not disease_name:
            raise HTTPException(status_code=400, detail="Cần cung cấp tên bệnh")
        translated_name = translate_disease_name(disease_name)
        search_result = search_disease_in_json(LOCAL_DATASET_PATH, translated_name)
        if not search_result:
            medline_result = search_medlineplus(disease_name)
            if medline_result:
                extracted_info = extract_medical_info(medline_result)
                if extracted_info:
                    search_result = [extracted_info]
        if not search_result:
            raise HTTPException(status_code=404, detail=f"Không tìm thấy thông tin cho bệnh: {disease_name}")
        logger.info(f"Tra cứu thông tin bệnh: {disease_name}")
        return JSONResponse(content={"disease_info": search_result}, status_code=200)
    except Exception as e:
        logger.error(f"Lỗi khi tra cứu thông tin bệnh: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi tra cứu thông tin bệnh: {str(e)}")

async def get_final_result(key: str):
    try:
        result = await get_result_by_key(key)
        result_diagnosis = result.get("result", [])
        if not result:
            raise HTTPException(status_code=404, detail="Không tìm thấy kết quả chẩn đoán")
        if not result_diagnosis:
            raise HTTPException (status_code=404,detail="Kết quả không tồn tại")
        return JSONResponse(content={ "diagnosis": result_diagnosis}, status_code=200)
    except Exception as e:
        logger.error(f"Lỗi khi lấy kết quả chẩn đoán: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi lấy kết quả chẩn đoán")