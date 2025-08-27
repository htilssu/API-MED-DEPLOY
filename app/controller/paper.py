from app.models.paper import Paper_Model
from fastapi import APIRouter, UploadFile as Upload, File, HTTPException
from app.db.mongo import db
from bson import ObjectId
from app.config.cloudinary_config import cloudinary
from io import BytesIO
from typing import List, Optional
from bson.errors import InvalidId

papers_collection = db["papers"]


def upload_image_bytes(image_bytes):
    response = cloudinary.uploader.upload(BytesIO(image_bytes))
    return response["secure_url"]

# Tối ưu hóa hàm upload
async def handle_upload(image: Optional[Upload]) -> Optional[str]:
    if image:
        result = upload_image_bytes(await image.read())
        return result if isinstance(result, str) else result.get("secure_url")
    return None


# Paper logic
async def create_paper(title: str, content: str, image: Upload = None,author: Optional[str] = None, authorImage: Optional[Upload] = None, authorDescription: Optional[str] = None, tags: Optional[List[str]] = None):
    image_url = await handle_upload(image)
 
    author_image_url = await handle_upload(authorImage) 
    paper_data = Paper_Model(title=title, content=content, mainImage=image_url,author=author,authorImage=author_image_url , authorDescription=authorDescription, tags=tags)
    # Kiểm tra tiêu đề và nội dung có tồn tại hay chưa
    existing_paper= await papers_collection.find_one({"title": title, "content": content})
    if existing_paper:
        raise HTTPException(status_code=400, detail="Bài viết với tiêu đề và nội dung này đã tồn tại.")
    # Chèn dữ liệu vào cơ sở dữ liệu
    result = await papers_collection.insert_one(paper_data.model_dump(by_alias=True, exclude_none=True))
    new_paper = await papers_collection.find_one({"_id": result.inserted_id})
    if new_paper:
        new_paper["_id"] = str(new_paper["_id"])
        return Paper_Model(**new_paper)
    
    return None

async def get_all_papers():
    papers = []
    async for paper in papers_collection.find():
        paper["_id"] = str(paper["_id"])
        papers.append(Paper_Model(**paper))
    return papers

async def get_paper_by_id(paper_id: str):
    paper = await papers_collection.find_one({"_id": ObjectId(paper_id)})
    if paper:
        paper["_id"] = str(paper["_id"])
        return Paper_Model(**paper)
    return None

async def get_papers_by_tag(tag: str):
    try:
        tag_id = ObjectId(tag)
    except InvalidId:
        raise HTTPException(status_code=400, detail="ID tag không hợp lệ.")
    papers = []
    async for paper in papers_collection.find({"tags": tag}):
        paper["_id"] = str(paper["_id"])
        papers.append(Paper_Model(**paper))
    return papers

async def update_paper(paper_id: str, title: str = None, content: str = None, image: Upload = None, author: Optional[str] = None, authorImage: Optional[Upload] = None, authorDescription: Optional[str] = None, tags: Optional[List[str]] = None):
    update_data = {k: v for k, v in [("title", title), ("content", content),("author",author),("authorDescription",authorDescription),("tags",tags)] if v is not None}
    
    image_url = await handle_upload(image)
    author_image_url = await handle_upload(authorImage)
    if author_image_url:
        update_data["authorImage"] = author_image_url
    if image_url:
        update_data["mainImage"] = image_url
    result = await papers_collection.update_one({"_id": ObjectId(paper_id)}, {"$set": update_data})
    return await get_paper_by_id(paper_id) if result.modified_count else None

async def delete_paper(paper_id: str):
    result = await papers_collection.delete_one({"_id": ObjectId(paper_id)})
    return result.deleted_count > 0

async def search_papers(query: str):
    """
    Tìm kiếm bài viết theo tiêu đề hoặc nội dung.
    """
    papers = []
    async for paper in papers_collection.find({
        "$or": [
            {"title": {"$regex": query, "$options": "i"}},
            {"content": {"$regex": query, "$options": "i"}}
        ]
    }):
        paper["_id"] = str(paper["_id"])
        papers.append(Paper_Model(**paper))
    return papers