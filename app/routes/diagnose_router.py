from app.controller.diagnose_controller import (create_diagnosis, get_diagnosis, get_user_diagnoses, delete_diagnosis)
from fastapi import APIRouter, UploadFile, File, Query, Body, Form, HTTPException
from typing import List, Optional
from fastapi.responses import JSONResponse
from app.config.cloudinary_config import cloudinary

from app.models.diagnoiseModel import CreateDiagnoseModel, DiagnoseModel

router = APIRouter()


# === DIAGNOSIS ===
@router.post("/create-diagnosis", response_model=DiagnoseModel, status_code=201)
async def create_diagnosis_route(diagnosis_data: CreateDiagnoseModel):
    try:
        diagnosis_id = await create_diagnosis(diagnosis_data)
        created = await get_diagnosis(diagnosis_id)
        return created
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/get-diagnosis/{diagnosis_id}", response_model=DiagnoseModel)
async def get_diagnosis_route(diagnosis_id: str):
    diagnosis = await get_diagnosis(diagnosis_id)
    if not diagnosis:
        raise HTTPException(status_code=404, detail="Diagnosis not found")
    return diagnosis


@router.get("/user-diagnoses/{user_id}", response_model=List[DiagnoseModel])
async def get_user_diagnoses_route(user_id: str):
    return await get_user_diagnoses(user_id)


@router.delete("/delete-diagnosis/{diagnosis_id}", response_model=dict)
async def delete_diagnosis_route(diagnosis_id: str):
    deleted = await delete_diagnosis(diagnosis_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Diagnosis not found")
    return {"message": "Diagnosis deleted successfully"}