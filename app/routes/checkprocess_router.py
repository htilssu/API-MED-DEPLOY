from app.controller.checkprocess_controller import create_check_process, get_check_process_by_user, get_check_process_by_id,track_check_process,delete_check_process
from fastapi import APIRouter, UploadFile, File, Query, Body, Form, HTTPException
from typing import List, Optional
from fastapi.responses import JSONResponse
from app.models.checkprocessModel import CheckProcessModel

router = APIRouter()


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