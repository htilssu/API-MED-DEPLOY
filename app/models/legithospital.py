from typing import List,Optional
from pydantic import BaseModel, Field, EmailStr
from bson import ObjectId
from datetime import datetime, date


class LegitHospitalModel(BaseModel):
    id: Optional[str] = Field(default=None, alias="_id")
    name: str
    address: str
    phone: Optional[str] = None
    img: Optional[str] = None
    yearEstablished: Optional[int] = None
    specialties: List[str]
    region: Optional[str] = None
    hospitalDescription: Optional[str] = None  # Mô tả bệnh viện (nếu có)
    rate: Optional[float] = None # Đánh giá bệnh viện (nếu có)

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
                "region": "Miền Nam",
                "hospitalDescription": "Bệnh viện chuyên khoa da liễu hàng đầu tại TP.HCM.",
                "rate": 4.5
            }
        }