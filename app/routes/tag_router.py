from app.controller.tag_controller import create_tag, get_all_tags, get_tag_by_id
from fastapi import APIRouter, UploadFile, File, Query, Body, Form, HTTPException
from typing import List, Optional
from fastapi.responses import JSONResponse
from app.models.tagModel import TagModel
router = APIRouter()


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
