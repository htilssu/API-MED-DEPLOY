from app.controller.tag import create_tag, get_all_tags, get_tag_by_id, update_tag,delete_tag
from fastapi import APIRouter, UploadFile, File, Query, Body, Form, HTTPException
from typing import List, Optional
from fastapi.responses import JSONResponse
from app.models.tag import TagModel
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

@router.put("/tag/{tag_id}", response_model=TagModel)
async def update_tag_route(
    tag_id: str,
    name: Optional[str] = Form(None, description="Tên mới của thẻ")
):
    """
    Cập nhật thông tin thẻ theo ID.
    """
    tag = await update_tag(tag_id=tag_id, name=name)
    return tag.model_dump(by_alias=True)


@router.delete("/tag/{tag_id}", status_code=204)
async def delete_tag_route(tag_id: str):
    """
    Xóa thẻ theo ID.
    """
    await delete_tag(tag_id)
    return JSONResponse(status_code=204, content=None)