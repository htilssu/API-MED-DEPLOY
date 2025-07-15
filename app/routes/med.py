from typing import Optional
from fastapi import APIRouter, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse
from app.controller.med import (
    start_diagnois,
    get_diagnosis_result,
    submit_discriminative_questions,
    get_discriminative_questions,
    knowledge,
    generate_disease_name,
)

router = APIRouter()

# === NHÓM: Tạo tên bệnh ===
@router.get("/generate-disease-name")
async def generate_disease_name_route(
    disease_name: str = Query(..., description="Tên bệnh cần tạo (tiếng Việt hoặc tiếng Anh)")
):
    """
    Tạo tên bệnh từ mô tả triệu chứng.
    """
    return generate_disease_name(disease_name)


# === NHÓM: Chẩn đoán bệnh ===
@router.post("/diagnosis/start", response_model=dict)
async def start_diagnosis_endpoint(
    image: UploadFile = File(...),
    user_id: Optional[str] = Query(None, description="ID của người dùng")
):
    """
    Bắt đầu quá trình chẩn đoán với ảnh và ID người dùng (nếu có).
    """
    return await start_diagnois(image=image, user_id=user_id)


@router.get("/diagnosis/result", response_model=dict)
async def fetch_result(key: str = Query(..., description="Key của chẩn đoán trong Redis")):
    """
    Lấy kết quả chẩn đoán từ Redis bằng key.
    """
    result = await get_diagnosis_result(key)
    return JSONResponse(content=result)


@router.get("/diagnosis/{key}", response_model=dict)
async def get_diagnosis_result_endpoint(key: str):
    """
    Lấy kết quả chẩn đoán dựa trên key.
    """
    return await get_diagnosis_result(key=key)


@router.get("/diagnosis/{key}/questions", response_model=dict)
async def get_discriminative_questions_endpoint(key: str):
    """
    Lấy danh sách câu hỏi phân biệt dựa trên key từ Redis.
    """
    return await get_discriminative_questions(key=key)


@router.post("/diagnosis/{key}/submit", response_model=dict)
async def submit_discriminative_questions_endpoint(
    key: str,
    user_answers: str = Query(..., description="Câu trả lời người dùng ở dạng JSON")
):
    """
    Gửi câu trả lời cho các câu hỏi phân biệt và nhận kết quả chẩn đoán cuối cùng.
    """
    return await submit_discriminative_questions(user_answers=user_answers, key=key)


# === NHÓM: Tri thức bệnh học ===
@router.get("/knowledge", response_model=dict)
async def get_disease_knowledge(
    disease_name: str = Query(..., description="Tên bệnh cần tra cứu (tiếng Việt hoặc tiếng Anh)")
):
    """
    Tra cứu thông tin bệnh từ dataset nội bộ hoặc MedlinePlus.
    """
    return await knowledge(disease_name)
