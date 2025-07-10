from fastapi import APIRouter, UploadFile, File,Query,Body,Form, HTTPException
from app.controller.med_controller import start_diagnois,get_diagnosis_result,get_differentiation_questions,submit_differation_questions,knowledge,submit_user_description,get_final_result,generate_disease_name
from fastapi.responses import JSONResponse
# from app.models.userModel import PostUserDescriptionModel
router = APIRouter()

from app.models.userModel import PostUserDescriptionModel,SubmitDifferentiationModel

@router.post("/start-diagnosis")
async def start_diagnosis_route(
    image: UploadFile = File(...),
    user_id: str = Form(None)
):
    return await start_diagnois(image=image, user_id=user_id)

@router.get("/diagnosis/result")
async def fetch_result(key: str = Query(...)):
    result = await get_diagnosis_result(key)
    return JSONResponse(content=result)

@router.get("/differentiation_questions")
async def get_questions(key: str = Query(...)):
    return await get_differentiation_questions(key)

@router.post("/submit-user-description")
async def submit_user_description_route(
    user_description:str= Body(..., description="Mô tả triệu chứng của người dùng"),
    key: str = Query(..., description="Key của kết quả đã lưu")
):
    """
    Gửi mô tả triệu chứng từ người dùng để cải thiện độ chính xác của chuẩn đoán.
    """
    return await submit_user_description(user_description=user_description, key=key)
     

@router.post("/submit-differentiation-questions")
async def submit_differentiation_route(
    key: str = Query(..., description="Key của kết quả đã lưu"),
    user_answers: dict = Body(..., description="Câu trả lời của người dùng cho câu hỏi phân biệt")
):
    """
    Gửi câu trả lời phân biệt từ người dùng để loại trừ nhãn không phù hợp.
    """
    await submit_differation_questions(user_answers=user_answers, key=key)
    return {"message": "Đã xử lý câu trả lời phân biệt thành công"}
    
    
@router.get("/knowledge")
async def get_disease_knowledge(
    disease_name: str = Query(..., description="Tên bệnh cần tra cứu (tiếng Việt hoặc tiếng Anh)")
):
    """
    Tra cứu thông tin bệnh từ dataset cục bộ hoặc MedlinePlus.
    """
    return await knowledge(disease_name)

@router.get("/final-diagnose")
async def get_final_diagnose(key: str):
    try:
        result = await get_final_result(key)
        return result  
    except Exception as e:
        raise HTTPException(status_code=500, detail="Lỗi không xác định")
    
@router.get("/generate-disease-name")
async def generate_disease_name_route(
    disease_name: str = Query(..., description="Tên bệnh cần tạo (tiếng Việt hoặc tiếng Anh)")
):
    """
    Tạo tên bệnh từ mô tả triệu chứng.
    """
    return await generate_disease_name(disease_name)