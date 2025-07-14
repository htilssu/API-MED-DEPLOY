from fastapi import APIRouter, UploadFile as Upload, File, HTTPException
from pydantic import BaseModel
import requests
from dotenv import load_dotenv
load_dotenv()
import os
import logging
import re
import json
import google.generativeai as genai
from typing import List, Dict
from geopy.geocoders import Nominatim


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
MAPBOX_ACCESS_TOKEN = os.getenv("mapbox_key")
logger = logging.getLogger(__name__)

def generate_bbox(lat:float, lng:float, delta=10):
    # ~10km quanh vị trí người dùng
    min_lng = lng - delta
    min_lat = lat - delta
    max_lng = lng + delta
    max_lat = lat + delta
    return f"{min_lng},{min_lat},{max_lng},{max_lat}"

async def search_pharmacies(lat:float,lng:float):
    bbox = generate_bbox(lat, lng)
    url = "https://api.mapbox.com/geocoding/v5/mapbox.places/pharmacy.json"
    params = {
        "proximity": f"{lng},{lat}",
        "bbox": bbox,
        "country": "VN",  # Giới hạn trong Việt Nam
        "access_token": MAPBOX_ACCESS_TOKEN,
        "limit": 10
    }

    response = requests.get(url, params=params)
    data = response.json()

    results = []
    for feature in data.get("features", []):
        results.append({
            "name": feature["text"],
            "address": feature.get("place_name", ""),
            "lat": feature["center"][1],
            "lng": feature["center"][0],
        })

    return results

def get_city_from_coordinates(lat: float, lon: float) -> str:
    geolocator = Nominatim(user_agent="geoapi_vi", timeout=10)
    location = geolocator.reverse((lat, lon), language="vi")  # ngôn ngữ tiếng Việt

    if location and location.raw and "address" in location.raw:
        address = location.raw["address"]
        # Ưu tiên lấy thành phố, nếu không có thì lấy quận, tỉnh
        return (
            address.get("city") or
            address.get("town") or
            address.get("village") or
            address.get("county") or
            address.get("state")
        )
    return "Không xác định được thành phố"


async def get_hospital_from_coordinates(lat: float, lon: float) -> List[Dict[str, str]]:
    city = get_city_from_coordinates(lat, lon)
    if not city:
        raise HTTPException(status_code=400, detail="Không xác định được thành phố từ tọa độ.")

    prompt = (
        f"Bạn là chuyên gia địa lý y tế tại Việt Nam. "
        f"Hãy liệt kê danh sách các bệnh viện uy tín chuyên chữa trị bệnh da liễu tại thành phố {city}. "
        "Chỉ trả về kết quả dạng JSON Array gồm các object có dạng:\n"
        "[{\"name\": \"Tên bệnh viện\", \"address\": \"Địa chỉ cụ thể\"}]\n"
        "Không cần giải thích, không thêm chú thích hoặc markdown."
    )

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        raw_text = response.text.strip()

        logger.debug(f"Gemini raw output:\n{raw_text}")

        # Làm sạch JSON nếu bị bọc trong markdown
        clean_text = re.sub(r"```(?:json)?|```", "", raw_text).strip()

        hospitals = json.loads(clean_text)
        if not isinstance(hospitals, list):
            raise ValueError("Phản hồi không phải là danh sách JSON.")

        logger.info(f"Tìm thấy {len(hospitals)} bệnh viện tại {city}")
        return hospitals

    except json.JSONDecodeError as e:
        logger.error(f"Lỗi JSON: {e}")
        raise HTTPException(status_code=500, detail="Phản hồi từ Gemini không đúng định dạng JSON.")
    except Exception as e:
        logger.exception("Lỗi khi tìm kiếm bệnh viện")
        raise HTTPException(status_code=500, detail=f"Lỗi khi tìm kiếm bệnh viện: {str(e)}")