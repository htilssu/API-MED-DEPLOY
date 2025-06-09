from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional
from bson import ObjectId
from datetime import date,datetime

# Không cần lớp PyObjectId nữa
class UserModel(BaseModel):
    id: Optional[str] = Field(alias="_id")  # Thay PyObjectId bằng str
    name: str
    email: EmailStr
    phone: Optional[str] = None
    password: Optional[str] = None
    dateOfBirth: date

    class Config:
        populate_by_name = True
        json_encoders = {
            ObjectId: str  # Chuyển ObjectId thành str khi trả về JSON 
        }
        json_schema_extra = {
            "example": {
                "name": "your name",
                "email": "yourEmail@example.com",
                "phone": "0123456789",
                "password": "yourPassword",
                "dateOfBirth": "2004-01-10"
            }
        }

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



# Model cho tạo người dùng (không cần _id)
class CreateUserModel(BaseModel):
    name: str
    email: EmailStr
    phone: Optional[str] = None
    password: Optional[str] = None
    dateOfBirth: Optional[str] = None

# Model lưu trữ thông tin chuẩn đoán (không cần _id)
class CreateDiagnoseModel(BaseModel):
    userId: str
    diseaseResult: List[str]  # Mảng kết quả chuẩn đoán
    date: Optional[datetime] = Field(default_factory=datetime.now)  # Ngày giờ chuẩn đoán
    

# Model cho đăng nhập
class LoginModel(BaseModel):
    email: EmailStr
    password: str