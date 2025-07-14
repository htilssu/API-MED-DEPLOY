from fastapi import APIRouter, UploadFile, File, Query, Body, Form, HTTPException
from typing import List, Optional
from fastapi.responses import JSONResponse
from app.controller.legithospital_controller import (create_legit_hospital, get_all_legit_hospitals, get_legit_hospital_by_id,
    update_legit_hospital, delete_legit_hospital, get_hospitals_by_specialty, add_specialty_to_hospital,
    remove_specialty_from_hospital)

from app.models.legithospitalModel import LegitHospitalModel

router = APIRouter()


# === LEGIT HOSPITAL ===
@router.post("/create-legit-hospital", status_code=201)
async def create_legit_hospital_route(
    name: str = Form(...),
    address: str = Form(...),
    phone: Optional[str] = Form(None),
    img: Optional[UploadFile] = File(None),
    yearEstablished: Optional[int] = Form(None),
    specialties: List[str] = Form(...),
    region: Optional[str] = Form(None),
    hospitalDescription: Optional[str] = Form(None),
    rate: Optional[float] = Form(5.0)  # Default rate is 5
):
    hospital = await create_legit_hospital(name, address, phone, img, yearEstablished, specialties, region, hospitalDescription,rate)
    if not hospital:
        raise HTTPException(status_code=400, detail="Không thể tạo bệnh viện")
    return hospital.model_dump(by_alias=True)

@router.put("/update-legit-hospital/{hospital_id}", response_model=dict)
async def update_legit_hospital_route(
    hospital_id: str,
    name: Optional[str] = Form(None),
    address: Optional[str] = Form(None),
    phone: Optional[str] = Form(None),
    img: Optional[UploadFile] = File(None),
    yearEstablished: Optional[int] = Form(None),
    specialties: Optional[List[str]] = Form(None),
    region: Optional[str] = Form(None),
    hospitalDescription: Optional[str] = Form(None),
    rate: Optional[float] = Form(5.0)  # Default rate is 5
):
    hospital = await update_legit_hospital(hospital_id, name, address, phone, img, yearEstablished, specialties, region, hospitalDescription, rate)
    if not hospital:
        raise HTTPException(status_code=404, detail="Bệnh viện không tồn tại hoặc không có thay đổi")
    return hospital.model_dump(by_alias=True)


@router.get("/legit-hospitals", response_model=List[LegitHospitalModel])
async def get_all_legit_hospitals_route():
    return await get_all_legit_hospitals()

@router.get("/legit-hospital/{hospital_id}", response_model=dict)
async def get_legit_hospital_by_id_route(hospital_id: str):
    hospital = await get_legit_hospital_by_id(hospital_id)
    if not hospital:
        raise HTTPException(status_code=404, detail="Bệnh viện không tồn tại")
    return hospital.__dict__

@router.get("/hospitals-by-specialty", response_model=List[LegitHospitalModel])
async def get_hospitals_by_specialty_route(specialty: str):
    return await get_hospitals_by_specialty(specialty)

@router.post("/add-specialty-to-hospital", response_model=dict)
async def add_specialty_to_hospital_route(
    hospital_id: str = Form(...),
    specialty: str = Form(...)
):
    modified = await add_specialty_to_hospital(hospital_id, specialty)
    if not modified:
        raise HTTPException(status_code=404, detail="Bệnh viện không tồn tại hoặc chuyên khoa đã có")
    return {"message": "Chuyên khoa đã được thêm thành công"}

@router.post("/remove-specialty-from-hospital", response_model=dict)
async def remove_specialty_from_hospital_route(
    hospital_id: str = Form(...),
    specialty: str = Form(...)
):
    modified = await remove_specialty_from_hospital(hospital_id, specialty)
    if not modified:
        raise HTTPException(status_code=404, detail="Bệnh viện không tồn tại hoặc chuyên khoa không có")
    return {"message": "Chuyên khoa đã được xóa thành công"}

@router.delete("/delete-legit-hospital/{hospital_id}", response_model=dict)
async def delete_legit_hospital_route(hospital_id: str):
    deleted = await delete_legit_hospital(hospital_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Bệnh viện không tồn tại")
    return {"message": "Bệnh viện đã được xóa thành công"}