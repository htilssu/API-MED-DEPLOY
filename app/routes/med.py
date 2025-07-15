from typing import Optional
from fastapi import APIRouter, UploadFile, File,Query,Body,Form, HTTPException
from app.controller.med import start_diagnois,get_diagnosis_result,submit_discriminative_questions,get_discriminative_questions, knowledge,generate_disease_name
from fastapi.responses import JSONResponse
# from app.models.userModel import PostUserDescriptionModel
router = APIRouter()



@router.post("/", response_model=dict)
async def start_diagnosis_endpoint(
    image: UploadFile = File(...),
    user_id: Optional[str] = Query(None, description="ID của người dùng")
):
    """
    Bắt đầu quá trình chẩn đoán với ảnh và ID người dùng.
    
    Args:
        image: Ảnh tải lên (jpg, jpeg, png).
        user_id: ID của người dùng (tùy chọn).
    
    Returns:
        JSONResponse chứa kết quả chẩn đoán.
    """
    return await start_diagnois(image=image, user_id=user_id)

@router.get("/diagnosis/result")
async def fetch_result(key: str = Query(...)):
    result = await get_diagnosis_result(key)
    return JSONResponse(content=result)

@router.get("/{key}", response_model=dict)
async def get_diagnosis_result_endpoint(key: str):
    """
    Lấy kết quả chẩn đoán dựa trên key.
    
    Args:
        key: Key để truy xuất kết quả từ Redis.
    
    Returns:
        JSONResponse chứa kết quả chẩn đoán.
    """
    return await get_diagnosis_result(key=key)

@router.get("/{key}/questions", response_model=dict)
async def get_discriminative_questions_endpoint(key: str):
    """
    Lấy danh sách câu hỏi phân biệt dựa trên key.
    
    Args:
        key: Key để truy xuất dữ liệu từ Redis.
    
    Returns:
        JSONResponse chứa danh sách câu hỏi phân biệt.
    """
    return await get_discriminative_questions(key=key)

@router.post("/{key}/submit", response_model=dict)
async def submit_discriminative_questions_endpoint(key: str, user_answers: str):
    """
    Gửi câu trả lời cho câu hỏi phân biệt và nhận chẩn đoán cuối cùng.
    
    Args:
        key: Key để truy xuất dữ liệu từ Redis.
        user_answers: Câu trả lời của người dùng dưới dạng chuỗi JSON.
    
    Returns:
        JSONResponse chứa chẩn đoán cuối cùng.
    """
    return await submit_discriminative_questions(user_answers=user_answers, key=key)
    
    
@router.get("/knowledge")
async def get_disease_knowledge(
    disease_name: str = Query(..., description="Tên bệnh cần tra cứu (tiếng Việt hoặc tiếng Anh)")
):
    """
    Tra cứu thông tin bệnh từ dataset cục bộ hoặc MedlinePlus.
    """
    return await knowledge(disease_name)

    
@router.get("/generate-disease-name")
async def generate_disease_name_route(
    disease_name: str = Query(..., description="Tên bệnh cần tạo (tiếng Việt hoặc tiếng Anh)")
):
    """
    Tạo tên bệnh từ mô tả triệu chứng.
    """
    return await generate_disease_name(disease_name)