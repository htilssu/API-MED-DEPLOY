from typing import Optional
from pydantic import BaseModel, Field
from bson import ObjectId
from datetime import datetime

#Model cho tạo bài báo 
class CreateNewsModel(BaseModel):
    title: str = Field(..., description="Tiêu đề bài báo")
    content: str = Field(..., description="Nội dung bài báo")
    date:datetime = Field(default_factory=datetime.now)  # Ngày giờ đăng bài


class Paper_Model(BaseModel):
    id: Optional[str] = Field(default=None,alias="_id")
    title: str
    mainImage: Optional[str] = None  # URL của ảnh chính (nếu có)
    content: str
    date: datetime = Field(default_factory=datetime.now)  # Ngày giờ đăng bài
    author: Optional[str] = None  # Tên tác giả (nếu có)
    authorImage: Optional[str] = None  # URL ảnh tác giả (nếu có)
    authorDescription: Optional[str] = None  # Mô tả tác giả (nếu có)
    #Liên kết với các tag
    tags: List[str] = Field(default_factory=list)  # Danh sách ID của các tag liên quan
    class Config:
        json_encoders = {
            ObjectId: str  # Chuyển ObjectId thành str khi trả về JSON
        }
        json_schema_extra = {
            "example": {
                "title": "Bài viết về bệnh da liễu",
                "content": "Nội dung bài viết...",
                "date": "2023-10-01T12:00:00",
                "mainImage": "https://example.com/image.jpg",
                "author": "Nguyễn Văn A",
                "authorImage": "https://example.com/author.jpg",
                "authorDescription": "Chuyên gia da liễu hàng đầu",
                "tags": ["60c72b2f9b1e8b001c8e4d3a", "60c72b2f9b1e8b001c8e4d3b"]  # Danh sách ID của các tag
            }
        }