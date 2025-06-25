from app.models.userModel import DiagnoseModel,Paper_Model, CheckProcessModel, LegitHospitalModel
from fastapi import APIRouter, UploadFile as Upload, File, HTTPException
from app.db.mongo import db
from bson import ObjectId
from dotenv import load_dotenv
from app.config.cloudinary_config import cloudinary
from io import BytesIO


def upload_image_bytes(image_bytes):
    response = cloudinary.uploader.upload(BytesIO(image_bytes))
    return response["secure_url"]

load_dotenv()

diagnoses_collection = db["diagnoses"]
papers_collection = db["papers"]
check_process_collection = db["check_process"]



async def create_diagnosis(diagnosis_data: DiagnoseModel):
    """
    Tạo một bản ghi chuẩn đoán mới.
    """
    diagnosis_data_dict = diagnosis_data.dict()
    diagnosis_data_dict["_id"] = ObjectId()  # Tạo ObjectId mới
    result = await diagnoses_collection.insert_one(diagnosis_data_dict)
    return str(result.inserted_id)  # Trả về ID của bản ghi đã tạo

async def get_diagnosis(diagnosis_id: str):
    """
    Lấy thông tin chuẩn đoán theo ID.
    """
    diagnosis = await diagnoses_collection.find_one({"_id": ObjectId(diagnosis_id)})
    if diagnosis:
        diagnosis["_id"] = str(diagnosis["_id"])  # Chuyển _id thành str
        return DiagnoseModel(**diagnosis)
    return None

async def get_user_diagnoses(user_id: str):
    """
    Lấy tất cả các bản ghi chuẩn đoán của người dùng theo userId.
    """
    diagnoses = []
    async for diagnosis in diagnoses_collection.find({" ": user_id}):
        diagnosis["_id"] = str(diagnosis["_id"])  # Chuyển _id thành str
        diagnoses.append(DiagnoseModel(**diagnosis))
    return diagnoses

async def delete_diagnosis(diagnosis_id: str):
    """
    Xóa một bản ghi chuẩn đoán theo ID.
    """
    result = await diagnoses_collection.delete_one({"_id": ObjectId(diagnosis_id)})
    if result.deleted_count == 0:
        return False
    return True
async def create_paper(title:str, content: str, image: Upload=None):
    """
    Tạo một bài báo mới.
    """
    image_url = None
    if image:
        upload_result = upload_image_bytes(image.file.read())
        image_url = upload_result["secure_url"]
    paper_data= Paper_Model(
        title=title,
        content=content,
        mainImageUrl=image_url
    )

    result = await papers_collection.insert_one(paper_data.dict())
    new_paper = await papers_collection.find_one({"_id": result.inserted_id})
    if new_paper:
        return Paper_Model(**new_paper)
    return None

#Hàm lấy tất cả bài báo
async def get_all_papers():
    """
    Lấy tất cả các bài báo.
    """
    papers = []
    async for paper in papers_collection.find():
        paper["_id"] = str(paper["_id"])  # Chuyển _id thành str
        papers.append(Paper_Model(**paper))
    return papers

#Hàm lấy bài báo theo ID
async def get_paper_by_id(paper_id: str):
    """
    Lấy thông tin bài báo theo ID.
    """
    paper = await papers_collection.find_one({"_id": ObjectId(paper_id)})
    if paper:
        paper["_id"] = str(paper["_id"])  # Chuyển _id thành str
        return Paper_Model(**paper)
    return None

#Hàm cập nhật bài báo
async def update_paper(paper_id: str, title: str = None, content: str = None, image: Upload = None):
    """
    Cập nhật thông tin bài báo.
    """
    update_data = {}
    if title:
        update_data["title"] = title
    if content:
        update_data["content"] = content
    if image:
        upload_result = upload_image_bytes(image.file.read())
        update_data["mainImageUrl"] = upload_result["secure_url"]

    result = await papers_collection.update_one({"_id": ObjectId(paper_id)}, {"$set": update_data})
    if result.modified_count > 0:
        return await get_paper_by_id(paper_id)
    return None

async def delete_paper(paper_id: str):
    """
    Xóa một bài báo theo ID.
    """
    result = await papers_collection.delete_one({"_id": ObjectId(paper_id)})
    return result.deleted_count > 0


async def create_check_process(user_id:str,image_urls: Upload):
    """
    Tạo một quá trình kiểm tra mới.
    """
    check_process_data= CheckProcessModel(
        userId=user_id,
        imageUrl= [upload_image_bytes(image_urls.file.read())]  # Giả sử chỉ có một ảnh
    )
    result = await check_process_collection.insert_one(check_process_data.dict())
    new_check_process = await check_process_collection.find_one({"_id": result.inserted_id})
    if new_check_process:
        new_check_process["_id"] = str(new_check_process["_id"])  # Chuyển _id thành str
        return CheckProcessModel(**new_check_process)
    return None

async def track_check_process(user_id: str,image_urls: Upload,checkProcess_id:str):
    """
    Thêm ảnh và trả về ảnh đầu tiên và mới nhất để xem sự thay đổi
    """
    imageURL=upload_image_bytes(image_urls.file.read())
    check_process = await check_process_collection.find_one({"_id": ObjectId(checkProcess_id), "userId": user_id})
    if not check_process:
        raise ValueError("Không tìm thấy quá trình kiểm tra cho người dùng này")
    # Thêm ảnh mới vào danh sách ảnh
    check_process["imageUrl"].append(imageURL)
    await check_process_collection.update_one({"_id": ObjectId(checkProcess_id)}, {"$set": {"imageUrl": check_process["imageUrl"]}})
    # Trả về ảnh đầu tiên và mới nhất
    first_image = check_process["imageUrl"][0] if check_process["imageUrl"] else None
    latest_image = check_process["imageUrl"][-1] if check_process["imageUrl"] else None
    return {
        "first_image": first_image,
        "latest_image": latest_image
    }
    
async def get_check_process_by_user(user_id: str):
    """
    Lấy tất cả các quá trình kiểm tra của người dùng theo userId.
    """
    check_processes = []
    async for check_process in check_process_collection.find({"userId": user_id}):
        check_process["_id"] = str(check_process["_id"])  # Chuyển _id thành str
        check_processes.append(CheckProcessModel(**check_process))
    return check_processes

async def get_check_process_by_id(check_process_id: str):
    """
    Lấy thông tin quá trình kiểm tra theo ID.
    """
    check_process = await check_process_collection.find_one({"_id": ObjectId(check_process_id)})
    if check_process:
        check_process["_id"] = str(check_process["_id"])  # Chuyển _id thành str
        return CheckProcessModel(**check_process)
    return None

async def delete_check_process(check_process_id: str):
    """
    Xóa một quá trình kiểm tra theo ID.
    """
    result = await check_process_collection.delete_one({"_id": ObjectId(check_process_id)})
    return result.deleted_count > 0

async def create_legit_hospital(name: str, address: str, phone: str = None, img: Upload = None, yearEstablished: int = None, specialties: list = [], region: str = None):
    """
    Tạo một bệnh viện uy tín
    """
    img_url = upload_image_bytes(img.file.read()) if img else None    
    hospital_data = LegitHospitalModel(
        name=name,
        address=address,
        phone=phone,
        imageUrl=img_url,
        yearEstablished=yearEstablished,
        specialties=specialties,
        region=region
    )
    result = await db["legit_hospitals"].insert_one(hospital_data.dict())
    new_hospital = await db["legit_hospitals"].find_one({"_id": result.inserted_id})
    if new_hospital:
        new_hospital["_id"] = str(new_hospital["_id"])  # Chuyển _id thành str
        return LegitHospitalModel(**new_hospital)
    return None

async def get_all_legit_hospitals():
    """
    Lấy tất cả các bệnh viện uy tín
    """
    hospitals = []
    async for hospital in db["legit_hospitals"].find():
        hospital["_id"] = str(hospital["_id"])  # Chuyển _id thành str
        hospitals.append(LegitHospitalModel(**hospital))
    return hospitals

async def get_legit_hospital_by_id(hospital_id: str):
    """
    Lấy thông tin bệnh viện uy tín theo ID.
    """
    hospital = await db["legit_hospitals"].find_one({"_id": ObjectId(hospital_id)})
    if hospital:
        hospital["_id"] = str(hospital["_id"])  # Chuyển _id thành str
        return LegitHospitalModel(**hospital)
    return None

async def update_legit_hospital(hospital_id: str, name: str = None, address: str = None, phone: str = None, img: Upload = None, yearEstablished: int = None, specialties: list = None, region: str = None):
    """
    Cập nhật thông tin bệnh viện uy tín.
    """
    update_data = {}
    if name:
        update_data["name"] = name
    if address:
        update_data["address"] = address
    if phone:
        update_data["phone"] = phone
    if img:
        upload_result = upload_image_bytes(img.file.read())
        update_data["imageUrl"] = upload_result["secure_url"]
    if yearEstablished is not None:
        update_data["yearEstablished"] = yearEstablished
    if specialties is not None:
        update_data["specialties"] = specialties
    if region is not None:
        update_data["region"] = region

    result = await db["legit_hospitals"].update_one({"_id": ObjectId(hospital_id)}, {"$set": update_data})
    if result.modified_count > 0:
        return await get_legit_hospital_by_id(hospital_id)
    return None

async def delete_legit_hospital(hospital_id: str):
    """
    Xóa một bệnh viện uy tín theo ID.
    """
    result = await db["legit_hospitals"].delete_one({"_id": ObjectId(hospital_id)})
    return result.deleted_count > 0

async def add_specialty_to_hospital(hospital_id: str, specialty: str):
    """
    Thêm một chuyên khoa vào bệnh viện uy tín.
    """
    result = await db["legit_hospitals"].update_one(
        {"_id": ObjectId(hospital_id)},
        {"$addToSet": {"specialties": specialty}}  # Sử dụng $addToSet để tránh trùng lặp
    )
    return result.modified_count > 0

async def remove_specialty_from_hospital(hospital_id: str, specialty: str):
    """
    Xóa một chuyên khoa khỏi bệnh viện uy tín.
    """
    result = await db["legit_hospitals"].update_one(
        {"_id": ObjectId(hospital_id)},
        {"$pull": {"specialties": specialty}}  # Sử dụng $pull để xóa chuyên khoa
    )
    return result.modified_count > 0

async def get_hospitals_by_specialty(specialty: str):
    """
    Lấy tất cả các bệnh viện uy tín theo chuyên khoa.
    """
    hospitals = []
    async for hospital in db["legit_hospitals"].find({"specialties": specialty}):
        hospital["_id"] = str(hospital["_id"])  # Chuyển _id thành str
        hospitals.append(LegitHospitalModel(**hospital))
    return hospitals