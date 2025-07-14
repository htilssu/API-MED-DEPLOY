from typing import List,Optional
from pydantic import BaseModel, Field
from bson import ObjectId
from datetime import datetime,date

#Model Thông tin chuẩn đoán
class DiagnoseModel(BaseModel):
    id: Optional[str] = Field(alias="_id")
    userId: str
    diseaseResult: List[str]  # Mảng kết quả chuẩn đoán
    date: datetime = Field(default_factory=datetime.now)  # Ngày giờ chuẩn đoán

    class Config:
        json_encoders = {
            ObjectId: str  # Chuyển ObjectId thành str khi trả về JSON
        }
        json_schema_extra = {
            "example": {
                "userId": "user_id_here",
                "diseaseResult": ["disease1", "disease2"],
                "date": "2023-10-01T12:00:00"
            }
        }


# Model lưu trữ thông tin chuẩn đoán (không cần _id)
class CreateDiagnoseModel(BaseModel):
    userId: str
    diseaseResult: List[str]  # Mảng kết quả chuẩn đoán
    date: Optional[datetime] = Field(default_factory=datetime.now)  # Ngày giờ chuẩn đoán


class PostUserDescriptionModel(BaseModel):
    user_description: str = Field(..., description="Mô tả của người dùng về triệu chứng")

class SubmitDifferentiationModel(BaseModel):
    description: str = Field(..., description="Mô tả triệu chứng của người dùng")