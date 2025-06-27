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

class Paper_Model(BaseModel):
    id: Optional[str] = Field(default=None,alias="_id")
    title: str
    mainImage: Optional[str] = None  # URL của ảnh chính (nếu có)
    content: str
    date: datetime = Field(default_factory=datetime.now)  # Ngày giờ đăng bài
    class Config:
        json_encoders = {
            ObjectId: str  # Chuyển ObjectId thành str khi trả về JSON
        }
        json_schema_extra = {
            "example": {
                "title": "Bài viết về bệnh da liễu",
                "content": "Nội dung bài viết...",
                "date": "2023-10-01T12:00:00"
            }
        }

class LegitHospitalModel(BaseModel):
    id: Optional[str] = Field(default=None, alias="_id")
    name: str
    address: str
    phone: Optional[str] = None
    img: Optional[str] = None
    yearEstablished: Optional[int] = None
    specialties: List[str]
    region: Optional[str] = None

    class Config:
        json_encoders = {
            ObjectId: str
        }
        json_schema_extra = {
            "example": {
                "name": "Bệnh viện Da liễu",
                "address": "123 Đường ABC, Quận 1, TP.HCM",
                "phone": "0123456789",
                "img": "http://example.com/hospital.jpg",
                "yearEstablished": 1990,
                "specialties": ["Da liễu", "Thẩm mỹ"],
                "region": "Miền Nam"
            }
        }
        
#Model cho tạo bài báo 
class CreateNewsModel(BaseModel):
    title: str = Field(..., description="Tiêu đề bài báo")
    content: str = Field(..., description="Nội dung bài báo")
    date:datetime = Field(default_factory=datetime.now)  # Ngày giờ đăng bài





# Model cho tạo người dùng (không cần _id)
class CreateUserModel(BaseModel):
    name: str = Field(..., description="Tên người dùng")
    email: EmailStr = Field(..., description="Email người dùng")
    phone: Optional[str] = Field(None, description="Số điện thoại người dùng")
    password: Optional[str] = Field(None, description="Mật khẩu người dùng")
    dateOfBirth: date = Field(..., description="Ngày sinh của người dùng")

# Model lưu trữ thông tin chuẩn đoán (không cần _id)
class CreateDiagnoseModel(BaseModel):
    userId: str
    diseaseResult: List[str]  # Mảng kết quả chuẩn đoán
    date: Optional[datetime] = Field(default_factory=datetime.now)  # Ngày giờ chuẩn đoán
    

# Model cho đăng nhập
class LoginModel(BaseModel):
    email: EmailStr = Field(..., description="Email người dùng để đăng nhập")
    password: str = Field(..., description="Mật khẩu người dùng để đăng nhập")

class PostUserDescriptionModel(BaseModel):
    user_description: str = Field(..., description="Mô tả của người dùng về triệu chứng")

class SubmitDifferentiationModel(BaseModel):
    description: str = Field(..., description="Mô tả triệu chứng của người dùng")