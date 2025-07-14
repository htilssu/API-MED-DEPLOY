from app.models.diagnoiseModel import DiagnoseModel
from fastapi import APIRouter, UploadFile as Upload, File, HTTPException
from app.db.mongo import db
from bson import ObjectId
from dotenv import load_dotenv
from app.config.cloudinary_config import cloudinary
from io import BytesIO
from typing import List, Optional
from bson.errors import InvalidId

diagnoses_collection = db["diagnoses"]


async def create_diagnosis(diagnosis_data: DiagnoseModel):
    """
    Tạo một bản ghi chuẩn đoán mới.
    """
    diagnosis_data_dict = diagnosis_data.dict()
    diagnosis_data_dict["_id"] = ObjectId()  # Tạo ObjectId mới
    result = await diagnoses_collection.insert_one(diagnosis_data_dict)
    return str(result.inserted_id)  # Trả về ID của bản ghi đã tạo

async def get_diagnosis(diagnosis_id: str):
    """
    Lấy thông tin chuẩn đoán theo ID.
    """
    diagnosis = await diagnoses_collection.find_one({"_id": ObjectId(diagnosis_id)})
    if diagnosis:
        diagnosis["_id"] = str(diagnosis["_id"])  # Chuyển _id thành str
        return DiagnoseModel(**diagnosis)
    return None

async def get_user_diagnoses(user_id: str):
    """
    Lấy tất cả các bản ghi chuẩn đoán của người dùng theo userId.
    """
    diagnoses = []
    async for diagnosis in diagnoses_collection.find({" ": user_id}):
        diagnosis["_id"] = str(diagnosis["_id"])  # Chuyển _id thành str
        diagnoses.append(DiagnoseModel(**diagnosis))
    return diagnoses

async def delete_diagnosis(diagnosis_id: str):
    """
    Xóa một bản ghi chuẩn đoán theo ID.
    """
    result = await diagnoses_collection.delete_one({"_id": ObjectId(diagnosis_id)})
    if result.deleted_count == 0:
        return False
    return True
