from fastapi import APIRouter, UploadFile, File,Query,Body,Form, HTTPException
from fastapi.responses import JSONResponse
from app.models.userModel import CreateDiagnoseModel, DiagnoseModel
from app.controller.user_controller import get_diagnosis, create_diagnosis, get_user_diagnoses, delete_diagnosis
router = APIRouter()

@router.post("/create_diagnosis", response_model=DiagnoseModel)
async def create_diagnosis_route(diagnosis_data: CreateDiagnoseModel):
    try:
        diagnosis_id = await create_diagnosis(diagnosis_data)
        created = await get_diagnosis(diagnosis_id)  # Lấy lại bản ghi đầy đủ
        return created
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/get-diagnosis/{diagnosis_id}",response_model=DiagnoseModel)
async def get_diagnosis_route(diagnosis_id: str):
    """
    Lấy thông tin chuẩn đoán theo ID.
    """
    diagnosis = await get_diagnosis(diagnosis_id)
    if not diagnosis:
        return JSONResponse(status_code=404, content={"message": "Diagnosis not found"})
    return diagnosis

@router.get("/user-diagnoses/{user_id}", response_model=list[DiagnoseModel])
async def get_user_diagnoses_route(user_id: str):
    """
    Lấy tất cả các bản ghi chuẩn đoán của người dùng theo userId.
    """
    diagnoses = await get_user_diagnoses(user_id)
    return diagnoses

@router.delete("/delete-diagnosis/{diagnosis_id}")
async def delete_diagnosis_route(diagnosis_id: str):
    """
    Xóa một bản ghi chuẩn đoán theo ID.
    """
    deleted = await delete_diagnosis(diagnosis_id)
    if not deleted:
        return JSONResponse(status_code=404, content={"message": "Diagnosis not found"})
    return {"message": "Diagnosis deleted successfully"}


