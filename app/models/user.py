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
    urlImage: Optional[str] = None  # Thêm trường urlImage nếu cần

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
                "dateOfBirth": "2004-01-10",
                "urlImage": "https://example.com/your_image.jpg"
            }
        }

# Model cho tạo người dùng (không cần _id)
class CreateUserModel(BaseModel):
    name: str = Field(..., description="Tên người dùng")
    email: EmailStr = Field(..., description="Email người dùng")
    phone: Optional[str] = Field(None, description="Số điện thoại người dùng")
    password: Optional[str] = Field(None, description="Mật khẩu người dùng")
    dateOfBirth: date = Field(..., description="Ngày sinh của người dùng")

# Model cho đăng nhập
class LoginModel(BaseModel):
    email: EmailStr = Field(..., description="Email người dùng để đăng nhập")
    password: str = Field(..., description="Mật khẩu người dùng để đăng nhập")