from app.models.legithospitalModel import LegitHospitalModel
from fastapi import APIRouter, UploadFile as Upload, File, HTTPException
from app.db.mongo import db
from bson import ObjectId
from dotenv import load_dotenv
from app.config.cloudinary_config import cloudinary
from io import BytesIO
from typing import List, Optional
from bson.errors import InvalidId


legit_hospitals_collection = db["legit_hospitals"]
def upload_image_bytes(image_bytes):
    response = cloudinary.uploader.upload(BytesIO(image_bytes))
    return response["secure_url"]

# Tối ưu hóa hàm upload
async def handle_upload(image: Optional[Upload]) -> Optional[str]:
    if image:
        result = upload_image_bytes(await image.read())
        return result if isinstance(result, str) else result.get("secure_url")
    return None


# Hospital logic
async def create_legit_hospital(name: str, address: str, phone: Optional[str] = None, img: Optional[Upload] = None, yearEstablished: Optional[int] = None, specialties: List[str] = [], region: Optional[str] = None,hospitalDescription: Optional[str] = None, rate: Optional[float] = None):
    img_url = await handle_upload(img)
    data = LegitHospitalModel(
        name=name,
        address=address,
        phone=phone,
        img=img_url,
        yearEstablished=yearEstablished,
        specialties=specialties,
        region=region,
        hospitalDescription=hospitalDescription,
        rate=rate
    )
    result = await legit_hospitals_collection.insert_one(data.model_dump(by_alias=True, exclude_none=True))
    new_doc = await legit_hospitals_collection.find_one({"_id": result.inserted_id})
    if new_doc:
        new_doc["_id"] = str(new_doc["_id"])
        return LegitHospitalModel(**new_doc)
    return None

async def get_all_legit_hospitals():
    hospitals = []
    async for hospital in legit_hospitals_collection.find():
        hospital["_id"] = str(hospital["_id"])
        hospitals.append(LegitHospitalModel(**hospital))
    return hospitals

async def get_legit_hospital_by_id(hospital_id: str):
    hospital = await legit_hospitals_collection.find_one({"_id": ObjectId(hospital_id)})
    if hospital:
        hospital["_id"] = str(hospital["_id"])
        return LegitHospitalModel(**hospital)
    return None

async def update_legit_hospital(hospital_id: str, name: str = None, address: str = None, phone: str = None, img: Upload = None, yearEstablished: int = None, specialties: list = None, region: str = None, hospitalDescription: Optional[str] = None, rate: Optional[float] = None):
    update_data = {k: v for k, v in [("name", name), ("address", address), ("phone", phone), ("yearEstablished", yearEstablished), ("specialties", specialties), ("region", region),("hospitalDescription",hospitalDescription),("rate",rate )] if v is not None}
    img_url = await handle_upload(img)
    if img_url:
        update_data["img"] = img_url
    result = await legit_hospitals_collection.update_one({"_id": ObjectId(hospital_id)}, {"$set": update_data})
    return await get_legit_hospital_by_id(hospital_id) if result.modified_count else None

async def delete_legit_hospital(hospital_id: str):
    result = await legit_hospitals_collection.delete_one({"_id": ObjectId(hospital_id)})
    return result.deleted_count > 0

async def add_specialty_to_hospital(hospital_id: str, specialty: str):
    result = await legit_hospitals_collection.update_one({"_id": ObjectId(hospital_id)}, {"$addToSet": {"specialties": specialty}})
    return result.modified_count > 0

async def remove_specialty_from_hospital(hospital_id: str, specialty: str):
    result = await legit_hospitals_collection.update_one({"_id": ObjectId(hospital_id)}, {"$pull": {"specialties": specialty}})
    return result.modified_count > 0

async def get_hospitals_by_specialty(specialty: str):
    hospitals = []
    async for hospital in legit_hospitals_collection.find({"specialties": specialty}):
        hospital["_id"] = str(hospital["_id"])
        hospitals.append(LegitHospitalModel(**hospital))
    return hospitals
