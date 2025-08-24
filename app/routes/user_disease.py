from fastapi import APIRouter, HTTPException, Query, Path
from typing import List, Optional
from app.controller.user_disease import (
    create_user_disease,
    get_user_disease,
    get_user_diseases,
    update_user_disease,
    delete_user_disease,
    get_user_disease_history,
    search_user_diseases,
    create_user_disease_from_ai_diagnosis
)
from app.models.user_disease import (
    UserDiseaseModel,
    CreateUserDiseaseModel,
    UpdateUserDiseaseModel,
    UserDiseaseHistoryModel,
    DiseaseStatus
)

router = APIRouter()

# === USER DISEASE MANAGEMENT ===

@router.post("/user-diseases", response_model=dict, status_code=201)
async def create_user_disease_route(disease_data: CreateUserDiseaseModel):
    """
    Tạo thông tin bệnh mới cho người dùng
    """
    try:
        disease_id = await create_user_disease(disease_data)
        return {
            "message": "Tạo thông tin bệnh thành công",
            "disease_id": disease_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/user-diseases/{disease_id}", response_model=UserDiseaseModel)
async def get_user_disease_route(
    disease_id: str = Path(..., description="ID của thông tin bệnh")
):
    """
    Lấy thông tin bệnh theo ID
    """
    disease = await get_user_disease(disease_id)
    if not disease:
        raise HTTPException(status_code=404, detail="Không tìm thấy thông tin bệnh")
    return disease

@router.get("/users/{user_id}/diseases", response_model=List[UserDiseaseModel])
async def get_user_diseases_route(
    user_id: str = Path(..., description="ID của người dùng"),
    status: Optional[DiseaseStatus] = Query(None, description="Lọc theo trạng thái bệnh")
):
    """
    Lấy tất cả thông tin bệnh của người dùng, có thể lọc theo trạng thái
    """
    return await get_user_diseases(user_id, status)

@router.put("/user-diseases/{disease_id}", response_model=UserDiseaseModel)
async def update_user_disease_route(
    disease_id: str = Path(..., description="ID của thông tin bệnh"),
    update_data: UpdateUserDiseaseModel = ...
):
    """
    Cập nhật thông tin bệnh của người dùng
    """
    updated_disease = await update_user_disease(disease_id, update_data)
    if not updated_disease:
        raise HTTPException(status_code=404, detail="Không tìm thấy thông tin bệnh")
    return updated_disease

@router.delete("/user-diseases/{disease_id}", response_model=dict)
async def delete_user_disease_route(
    disease_id: str = Path(..., description="ID của thông tin bệnh")
):
    """
    Xóa thông tin bệnh của người dùng
    """
    deleted = await delete_user_disease(disease_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Không tìm thấy thông tin bệnh")
    return {"message": "Xóa thông tin bệnh thành công"}

@router.get("/users/{user_id}/disease-history", response_model=UserDiseaseHistoryModel)
async def get_user_disease_history_route(
    user_id: str = Path(..., description="ID của người dùng")
):
    """
    Lấy lịch sử bệnh tổng quan của người dùng
    """
    return await get_user_disease_history(user_id)

@router.get("/users/{user_id}/diseases/search", response_model=List[UserDiseaseModel])
async def search_user_diseases_route(
    user_id: str = Path(..., description="ID của người dùng"),
    q: str = Query(..., description="Từ khóa tìm kiếm (tên bệnh, triệu chứng, mô tả)")
):
    """
    Tìm kiếm bệnh của người dùng theo tên bệnh hoặc triệu chứng
    """
    return await search_user_diseases(user_id, q)

# === AI INTEGRATION ===

@router.post("/users/{user_id}/diseases/from-ai-diagnosis", response_model=dict, status_code=201)
async def create_disease_from_ai_diagnosis_route(
    user_id: str = Path(..., description="ID của người dùng"),
    diagnosis_key: str = Query(..., description="Key Redis của quá trình chẩn đoán AI"),
    disease_names: List[str] = Query(..., description="Danh sách tên bệnh từ AI"),
    confidence: Optional[float] = Query(None, description="Độ tin cậy của AI (0-1)")
):
    """
    Tạo thông tin bệnh từ kết quả chẩn đoán AI
    """
    try:
        disease_id = await create_user_disease_from_ai_diagnosis(
            user_id, diagnosis_key, disease_names, confidence
        )
        return {
            "message": "Lưu kết quả chẩn đoán AI thành công",
            "disease_id": disease_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# === DISEASE STATUS MANAGEMENT ===

@router.patch("/user-diseases/{disease_id}/status", response_model=UserDiseaseModel)
async def update_disease_status_route(
    disease_id: str = Path(..., description="ID của thông tin bệnh"),
    status: DiseaseStatus = Query(..., description="Trạng thái mới của bệnh")
):
    """
    Cập nhật trạng thái bệnh của người dùng
    """
    update_data = UpdateUserDiseaseModel(status=status)
    updated_disease = await update_user_disease(disease_id, update_data)
    if not updated_disease:
        raise HTTPException(status_code=404, detail="Không tìm thấy thông tin bệnh")
    return updated_disease

# === DISEASE STATISTICS ===

@router.get("/users/{user_id}/disease-stats", response_model=dict)
async def get_user_disease_stats_route(
    user_id: str = Path(..., description="ID của người dùng")
):
    """
    Lấy thống kê bệnh của người dùng
    """
    history = await get_user_disease_history(user_id)
    
    # Thống kê theo tháng gần nhất
    from datetime import datetime, timedelta
    from collections import defaultdict
    
    now = datetime.now()
    six_months_ago = now - timedelta(days=180)
    
    monthly_stats = defaultdict(int)
    recent_diseases = [d for d in history.diseases if d.diagnosisDate >= six_months_ago]
    
    for disease in recent_diseases:
        month_key = disease.diagnosisDate.strftime("%Y-%m")
        monthly_stats[month_key] += 1
    
    return {
        "total_diseases": history.totalDiseases,
        "active_diseases": history.activeDiseases,
        "recovered_diseases": history.recoveredDiseases,
        "chronic_diseases": history.chronicDiseases,
        "recent_diseases_count": len(recent_diseases),
        "monthly_diagnosis_trend": dict(monthly_stats),
        "last_updated": history.lastUpdated
    }