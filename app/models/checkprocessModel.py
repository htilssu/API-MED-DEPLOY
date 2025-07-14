from typing import List, Optional
from pydantic import BaseModel, Field
from bson import ObjectId

class CheckProcessModel(BaseModel):
    id: Optional[str] = Field(default=None, alias="_id")  # ✅ sửa chỗ này
    userId: str
    imageUrl: List[str]

    class Config:
        populate_by_name = True
        json_encoders = {ObjectId: str}
        json_schema_extra = {
            "example": {
                "userId": "123",
                "imageUrl": [
                    "https://res.cloudinary.com/.../abc.jpg",
                    "https://res.cloudinary.com/.../xyz.jpg"
                ]
            }
        }