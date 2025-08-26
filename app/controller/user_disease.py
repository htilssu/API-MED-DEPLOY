from app.models.user_disease import (
    UserDiseaseModel, 
    CreateUserDiseaseModel, 
    UpdateUserDiseaseModel,
    UserDiseaseHistoryModel,
    DiseaseStatus
)
from fastapi import HTTPException
from app.db.mongo import db
from bson import ObjectId
from typing import List, Optional
from datetime import datetime
from bson.errors import InvalidId

user_diseases_collection = db["user_diseases"]

async def create_user_disease(disease_data: CreateUserDiseaseModel) -> str:
    """
    Tạo thông tin bệnh mới cho người dùng
    """
    try:
        disease_data_dict = disease_data.model_dump()
        disease_data_dict["_id"] = ObjectId()
        disease_data_dict["createdAt"] = datetime.now()
        disease_data_dict["updatedAt"] = datetime.now()
        
        result = await user_diseases_collection.insert_one(disease_data_dict)
        return str(result.inserted_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi tạo thông tin bệnh: {str(e)}")

async def get_user_disease(disease_id: str) -> Optional[UserDiseaseModel]:
    """
    Lấy thông tin bệnh theo ID
    """
    try:
        disease = await user_diseases_collection.find_one({"_id": ObjectId(disease_id)})
        if disease:
            disease["_id"] = str(disease["_id"])
            return UserDiseaseModel(**disease)
        return None
    except InvalidId:
        raise HTTPException(status_code=400, detail="ID bệnh không hợp lệ")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi lấy thông tin bệnh: {str(e)}")

async def get_user_diseases(user_id: str, status: Optional[DiseaseStatus] = None) -> List[UserDiseaseModel]:
    """
    Lấy tất cả thông tin bệnh của người dùng, có thể lọc theo trạng thái
    """
    try:
        query = {"userId": user_id}
        if status:
            query["status"] = status.value
            
        diseases = []
        async for disease in user_diseases_collection.find(query).sort("diagnosisDate", -1):
            disease["_id"] = str(disease["_id"])
            diseases.append(UserDiseaseModel(**disease))
        return diseases
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi lấy danh sách bệnh: {str(e)}")

async def update_user_disease(disease_id: str, update_data: UpdateUserDiseaseModel) -> Optional[UserDiseaseModel]:
    """
    Cập nhật thông tin bệnh của người dùng
    """
    try:
        update_dict = {k: v for k, v in update_data.model_dump().items() if v is not None}
        update_dict["updatedAt"] = datetime.now()
        
        result = await user_diseases_collection.update_one(
            {"_id": ObjectId(disease_id)},
            {"$set": update_dict}
        )
        
        if result.matched_count == 0:
            return None
            
        return await get_user_disease(disease_id)
    except InvalidId:
        raise HTTPException(status_code=400, detail="ID bệnh không hợp lệ")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi cập nhật thông tin bệnh: {str(e)}")

async def delete_user_disease(disease_id: str) -> bool:
    """
    Xóa thông tin bệnh của người dùng
    """
    try:
        result = await user_diseases_collection.delete_one({"_id": ObjectId(disease_id)})
        return result.deleted_count > 0
    except InvalidId:
        raise HTTPException(status_code=400, detail="ID bệnh không hợp lệ")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xóa thông tin bệnh: {str(e)}")

async def get_user_disease_history(user_id: str) -> UserDiseaseHistoryModel:
    """
    Lấy lịch sử bệnh tổng quan của người dùng
    """
    try:
        diseases = await get_user_diseases(user_id)
        
        total_diseases = len(diseases)
        active_diseases = len([d for d in diseases if d.status == DiseaseStatus.ACTIVE])
        recovered_diseases = len([d for d in diseases if d.status == DiseaseStatus.RECOVERED])
        chronic_diseases = len([d for d in diseases if d.status == DiseaseStatus.CHRONIC])
        
        return UserDiseaseHistoryModel(
            userId=user_id,
            totalDiseases=total_diseases,
            activeDiseases=active_diseases,
            recoveredDiseases=recovered_diseases,
            chronicDiseases=chronic_diseases,
            diseases=diseases,
            lastUpdated=datetime.now()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi lấy lịch sử bệnh: {str(e)}")

async def search_user_diseases(user_id: str, search_term: str) -> List[UserDiseaseModel]:
    """
    Tìm kiếm bệnh của người dùng theo tên bệnh hoặc triệu chứng
    """
    try:
        query = {
            "userId": user_id,
            "$or": [
                {"diseaseName": {"$regex": search_term, "$options": "i"}},
                {"symptoms": {"$regex": search_term, "$options": "i"}},
                {"description": {"$regex": search_term, "$options": "i"}}
            ]
        }
        
        diseases = []
        async for disease in user_diseases_collection.find(query).sort("diagnosisDate", -1):
            disease["_id"] = str(disease["_id"])
            diseases.append(UserDiseaseModel(**disease))
        return diseases
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi tìm kiếm bệnh: {str(e)}")

async def create_user_disease_from_ai_diagnosis(user_id: str, diagnosis_key: str, disease_names: List[str], confidence: float = None) -> str:
    """
    Tạo thông tin bệnh từ kết quả chẩn đoán AI
    """
    try:
        # Lấy bệnh chính (bệnh đầu tiên trong danh sách)
        primary_disease = disease_names[0] if disease_names else "Không xác định"
        
        disease_data = CreateUserDiseaseModel(
            userId=user_id,
            diseaseName=primary_disease,
            symptoms=disease_names[1:] if len(disease_names) > 1 else [],  # Các bệnh khác có thể là triệu chứng
            aiDiagnosis=True,
            aiConfidence=confidence,
            diagnosisKey=diagnosis_key,
            description=f"Chẩn đoán AI với độ tin cậy {confidence:.2f}" if confidence else "Chẩn đoán bằng AI"
        )
        
        return await create_user_disease(disease_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi tạo bệnh từ AI: {str(e)}")