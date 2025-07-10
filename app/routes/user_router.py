from fastapi import APIRouter, UploadFile, File, Query, Body, Form, HTTPException
from typing import List, Optional
from fastapi.responses import JSONResponse
from app.config.cloudinary_config import cloudinary

from app.models.userModel import (
    CreateDiagnoseModel,
    DiagnoseModel,
    Paper_Model,
    CheckProcessModel,
    LegitHospitalModel,
    TagModel
)

def upload_image(image_path):
    response = cloudinary.uploader.upload(image_path)
    return response["secure_url"]

from app.controller.user_controller import (
    get_diagnosis, create_diagnosis, get_user_diagnoses, delete_diagnosis,
    create_legit_hospital, get_all_legit_hospitals, get_legit_hospital_by_id,
    get_hospitals_by_specialty, add_specialty_to_hospital, remove_specialty_from_hospital,
    delete_legit_hospital, get_check_process_by_id, get_check_process_by_user,
    create_check_process, track_check_process, delete_check_process,
    create_paper, get_all_papers, get_paper_by_id, update_paper, delete_paper,update_legit_hospital,
    get_papers_by_tag,get_all_tags,get_tag_by_id,create_tag
)

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


# === PAPERS ===
@router.post("/create-paper", response_model=dict, status_code=201)
async def create_paper_route(
    title: str = Form(...),
    content: str = Form(...),
    image: Optional[UploadFile] = File(None),
    author: Optional[str] = Form(None),
    authorImage: Optional[UploadFile] = File(None),
    authorDescription: Optional[str] = Form(None),
    tags: Optional[List[str]] = Form(None)
):
    paper = await create_paper(title, content, image,author, authorImage, authorDescription, tags)
    if not paper:
        raise HTTPException(status_code=400, detail="Không thể tạo bài báo")
    return paper.__dict__

@router.get("/papers", response_model=List[Paper_Model])
async def get_all_papers_route():
    return await get_all_papers()

@router.get("/papers-by-tag", response_model=List[Paper_Model])
async def get_papers_by_tag_route(tag: str = Query(..., description="Thẻ để lọc bài báo")):
    papers = await get_papers_by_tag(tag)
    if not papers:
        raise HTTPException(status_code=404, detail="Không tìm thấy bài báo với thẻ này")
    return papers

@router.get("/paper/{paper_id}", response_model=dict)
async def get_paper_by_id_route(paper_id: str):
    paper = await get_paper_by_id(paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Bài báo không tồn tại")
    return paper.__dict__

@router.post("/update-paper/{paper_id}", response_model=dict)
async def update_paper_route(
    paper_id: str,
    title: Optional[str] = Form(None),
    content: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    author: Optional[str] = Form(None),
    authorImage: Optional[UploadFile] = File(None),
    authorDescription: Optional[str] = Form(None),
    tags: Optional[List[str]] = Form(None)
):
    paper = await update_paper(paper_id, title, content, image, author, authorImage, authorDescription, tags)
    if not paper:
        raise HTTPException(status_code=404, detail="Bài báo không tồn tại hoặc không có thay đổi")
    return paper.__dict__

@router.delete("/delete-paper/{paper_id}", response_model=dict)
async def delete_paper_route(paper_id: str):
    deleted = await delete_paper(paper_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Bài báo không tồn tại")
    return {"message": "Bài báo đã được xóa thành công"}

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

# === CHECK PROCESS ===
@router.get("/check-process/{process_id}", response_model=dict)
async def get_check_process_by_id_route(process_id: str):
    process = await get_check_process_by_id(process_id)
    if not process:
        raise HTTPException(status_code=404, detail="Quá trình kiểm tra da không tồn tại")
    return process.__dict__

@router.get("/user-check-process/{user_id}", response_model=List[CheckProcessModel])
async def get_check_process_by_user_route(user_id: str):
    return await get_check_process_by_user(user_id)

@router.post("/create-check-process", response_model=dict, status_code=201)
async def create_check_process_route(
    user_id: str = Form(...),
    image: UploadFile = File(...)
):
    process = await create_check_process(user_id, image)
    if not process:
        raise HTTPException(status_code=400, detail="Không thể tạo quá trình kiểm tra da")
    return process.__dict__

@router.post("/track-check-process/{process_id}", response_model=dict)
async def track_check_process_route(
    process_id: str,
    image: UploadFile = File(...),
    user_id: str = Form(...)
):
    try:
        result = await track_check_process(user_id, image, process_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.delete("/delete-check-process/{process_id}", response_model=dict)
async def delete_check_process_route(process_id: str):
    deleted = await delete_check_process(process_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Quá trình kiểm tra da không tồn tại")
    return {"message": "Quá trình kiểm tra da đã được xóa thành công"}

@router.post("/create-tag", response_model=TagModel, status_code=201)
async def create_tag_route(
    name: str = Form(..., description="Tên của thẻ")
):
    tag = await create_tag(name)
    if not tag:
        raise HTTPException(status_code=400, detail="Không thể tạo thẻ")
    return tag.model_dump(by_alias=True)

@router.get("/tags", response_model=List[TagModel])
async def get_all_tags_route():
    return await get_all_tags()

@router.get("/tag/{tag_id}", response_model=TagModel)
async def get_tag_by_id_route(tag_id: str):
    tag = await get_tag_by_id(tag_id)
    if not tag:
        raise HTTPException(status_code=404, detail="Thẻ không tồn tại")
    return tag.model_dump(by_alias=True)

