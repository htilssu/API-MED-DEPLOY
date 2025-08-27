from app.models.tag import TagModel
from fastapi import APIRouter, UploadFile as Upload, File, HTTPException
from app.db.mongo import db
from bson import ObjectId
from app.config.cloudinary_config import cloudinary
from io import BytesIO
from typing import List, Optional
from bson.errors import InvalidId

tags_collection = db["tags"]


async def create_tag(name: str):
    """
    Tạo một thẻ mới.
    """
    existing_tag = await tags_collection.find_one({"name": name})
    if existing_tag:
        raise HTTPException(status_code=400, detail="Thẻ đã tồn tại")
    tag_data = TagModel(name=name)
    result = await tags_collection.insert_one(tag_data.model_dump(by_alias=True, exclude_none=True
    ))
    new_tag = await tags_collection.find_one({"_id": result.inserted_id})
    if new_tag:
        new_tag["_id"] = str(new_tag["_id"])  # Chuyển ObjectId thành str
        return TagModel(**new_tag)
    return None

async def get_all_tags():
    """
    Lấy tất cả các thẻ.
    """
    tags = []
    async for tag in tags_collection.find():
        tag["_id"] = str(tag["_id"])  # Chuyển ObjectId thành str
        tags.append(TagModel(**tag))
    return tags

async def get_tag_by_id(tag_id: str):
    """
    Lấy thông tin thẻ theo ID.
    """
    tag = await tags_collection.find_one({"_id": ObjectId(tag_id)})
    if tag:
        tag["_id"] = str(tag["_id"])  # Chuyển ObjectId thành str
        return TagModel(**tag)
    return None

async def update_tag(tag_id: str, name: Optional[str] = None):
    """
    Cập nhật thông tin thẻ.
    """
    if not tag_id:
        raise HTTPException(status_code=400, detail="ID không được để trống")

    update_data = {}
    if name:
        update_data["name"] = name

    if not update_data:
        raise HTTPException(status_code=400, detail="Không có dữ liệu để cập nhật")

    result = await tags_collection.update_one(
        {"_id": ObjectId(tag_id)}, {"$set": update_data}
    )

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Thẻ không tồn tại")

    return await get_tag_by_id(tag_id)

async def delete_tag(tag_id: str):
    """
    Xóa thẻ theo ID.
    """
    if not tag_id:
        raise HTTPException(status_code=400, detail="ID thẻ không được để trống")

    result = await tags_collection.delete_one({"_id": ObjectId(tag_id)})

    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Thẻ không tồn tại")