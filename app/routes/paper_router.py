from app.controller.paper_controller import create_paper, get_paper_by_id, get_all_papers, update_paper, delete_paper, get_papers_by_tag,search_papers
from fastapi import APIRouter, UploadFile, File, Query, Body, Form, HTTPException
from typing import List, Optional
from fastapi.responses import JSONResponse
from app.models.paperModel import Paper_Model


router = APIRouter()


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

