from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from bson import ObjectId
from datetime import datetime, date
from enum import Enum

class DiseaseStatus(str, Enum):
    """Trạng thái của bệnh"""
    ACTIVE = "active"           # Đang mắc bệnh
    RECOVERED = "recovered"     # Đã khỏi bệnh
    CHRONIC = "chronic"         # Bệnh mãn tính
    UNDER_TREATMENT = "under_treatment"  # Đang điều trị

class SeverityLevel(str, Enum):
    """Mức độ nghiêm trọng"""
    MILD = "mild"         # Nhẹ
    MODERATE = "moderate" # Vừa
    SEVERE = "severe"     # Nghiêm trọng
    CRITICAL = "critical" # Nguy hiểm

class UserDiseaseModel(BaseModel):
    """Model lưu trữ thông tin bệnh của người dùng chi tiết"""
    id: Optional[str] = Field(alias="_id")
    userId: str = Field(..., description="ID của người dùng")
    
    # Thông tin chẩn đoán cơ bản
    diseaseName: str = Field(..., description="Tên bệnh")
    diseaseCode: Optional[str] = Field(None, description="Mã bệnh (ICD-10)")
    diagnosisDate: datetime = Field(default_factory=datetime.now, description="Ngày chẩn đoán")
    
    # Trạng thái và mức độ
    status: DiseaseStatus = Field(default=DiseaseStatus.ACTIVE, description="Trạng thái bệnh")
    severity: Optional[SeverityLevel] = Field(None, description="Mức độ nghiêm trọng")
    
    # Triệu chứng và mô tả
    symptoms: List[str] = Field(default=[], description="Danh sách triệu chứng")
    description: Optional[str] = Field(None, description="Mô tả chi tiết về bệnh")
    
    # Thông tin điều trị
    treatment: Optional[str] = Field(None, description="Phương pháp điều trị")
    medications: List[str] = Field(default=[], description="Danh sách thuốc đang sử dụng")
    
    # Thông tin bác sĩ và cơ sở y tế
    doctorName: Optional[str] = Field(None, description="Tên bác sĩ chẩn đoán")
    hospitalName: Optional[str] = Field(None, description="Tên bệnh viện/phòng khám")
    
    # Thông tin AI (nếu được chẩn đoán bằng AI)
    aiDiagnosis: Optional[bool] = Field(False, description="Có phải chẩn đoán bằng AI không")
    aiConfidence: Optional[float] = Field(None, description="Độ tin cậy của AI (0-1)")
    diagnosisKey: Optional[str] = Field(None, description="Key Redis của quá trình chẩn đoán")
    
    # Ghi chú và theo dõi
    notes: Optional[str] = Field(None, description="Ghi chú thêm")
    followUpDate: Optional[datetime] = Field(None, description="Ngày tái khám")
    
    # Metadata
    createdAt: datetime = Field(default_factory=datetime.now, description="Ngày tạo")
    updatedAt: Optional[datetime] = Field(None, description="Ngày cập nhật")

    class Config:
        json_encoders = {
            ObjectId: str
        }
        json_schema_extra = {
            "example": {
                "userId": "user123",
                "diseaseName": "Viêm da cơ địa",
                "diseaseCode": "L20.9",
                "diagnosisDate": "2023-10-01T10:00:00",
                "status": "active",
                "severity": "moderate",
                "symptoms": ["Ngứa", "Đỏ da", "Khô da"],
                "description": "Viêm da cơ địa vùng khuỷu tay",
                "treatment": "Dùng kem dưỡng ẩm và thuốc kháng histamin",
                "medications": ["Cetirizine 10mg", "Kem hydrocortisone"],
                "doctorName": "BS. Nguyễn Văn A",
                "hospitalName": "Bệnh viện Da liễu TP.HCM",
                "aiDiagnosis": True,
                "aiConfidence": 0.85,
                "notes": "Cần tránh tiếp xúc với chất gây dị ứng"
            }
        }

class CreateUserDiseaseModel(BaseModel):
    """Model tạo thông tin bệnh mới cho người dùng"""
    userId: str = Field(..., description="ID của người dùng")
    diseaseName: str = Field(..., description="Tên bệnh")
    diseaseCode: Optional[str] = Field(None, description="Mã bệnh (ICD-10)")
    diagnosisDate: Optional[datetime] = Field(default_factory=datetime.now, description="Ngày chẩn đoán")
    
    status: DiseaseStatus = Field(default=DiseaseStatus.ACTIVE, description="Trạng thái bệnh")
    severity: Optional[SeverityLevel] = Field(None, description="Mức độ nghiêm trọng")
    
    symptoms: List[str] = Field(default=[], description="Danh sách triệu chứng")
    description: Optional[str] = Field(None, description="Mô tả chi tiết về bệnh")
    
    treatment: Optional[str] = Field(None, description="Phương pháp điều trị")
    medications: List[str] = Field(default=[], description="Danh sách thuốc đang sử dụng")
    
    doctorName: Optional[str] = Field(None, description="Tên bác sĩ chẩn đoán")
    hospitalName: Optional[str] = Field(None, description="Tên bệnh viện/phòng khám")
    
    aiDiagnosis: Optional[bool] = Field(False, description="Có phải chẩn đoán bằng AI không")
    aiConfidence: Optional[float] = Field(None, description="Độ tin cậy của AI (0-1)")
    diagnosisKey: Optional[str] = Field(None, description="Key Redis của quá trình chẩn đoán")
    
    notes: Optional[str] = Field(None, description="Ghi chú thêm")
    followUpDate: Optional[datetime] = Field(None, description="Ngày tái khám")

class UpdateUserDiseaseModel(BaseModel):
    """Model cập nhật thông tin bệnh của người dùng"""
    diseaseName: Optional[str] = Field(None, description="Tên bệnh")
    diseaseCode: Optional[str] = Field(None, description="Mã bệnh (ICD-10)")
    
    status: Optional[DiseaseStatus] = Field(None, description="Trạng thái bệnh")
    severity: Optional[SeverityLevel] = Field(None, description="Mức độ nghiêm trọng")
    
    symptoms: Optional[List[str]] = Field(None, description="Danh sách triệu chứng")
    description: Optional[str] = Field(None, description="Mô tả chi tiết về bệnh")
    
    treatment: Optional[str] = Field(None, description="Phương pháp điều trị")
    medications: Optional[List[str]] = Field(None, description="Danh sách thuốc đang sử dụng")
    
    doctorName: Optional[str] = Field(None, description="Tên bác sĩ chẩn đoán")
    hospitalName: Optional[str] = Field(None, description="Tên bệnh viện/phòng khám")
    
    notes: Optional[str] = Field(None, description="Ghi chú thêm")
    followUpDate: Optional[datetime] = Field(None, description="Ngày tái khám")
    
    updatedAt: datetime = Field(default_factory=datetime.now, description="Ngày cập nhật")

class UserDiseaseHistoryModel(BaseModel):
    """Model lịch sử bệnh của người dùng"""
    userId: str
    totalDiseases: int = Field(description="Tổng số bệnh")
    activeDiseases: int = Field(description="Số bệnh đang mắc")
    recoveredDiseases: int = Field(description="Số bệnh đã khỏi")
    chronicDiseases: int = Field(description="Số bệnh mãn tính")
    diseases: List[UserDiseaseModel] = Field(description="Danh sách tất cả bệnh")
    lastUpdated: datetime = Field(default_factory=datetime.now)