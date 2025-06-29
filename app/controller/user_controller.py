from app.models.userModel import DiagnoseModel,Paper_Model, CheckProcessModel, LegitHospitalModel
from fastapi import APIRouter, UploadFile as Upload, File, HTTPException
from app.db.mongo import db
from bson import ObjectId
from dotenv import load_dotenv
from app.config.cloudinary_config import cloudinary
from io import BytesIO
from typing import List, Optional


def upload_image_bytes(image_bytes):
    response = cloudinary.uploader.upload(BytesIO(image_bytes))
    return response["secure_url"]

load_dotenv()

diagnoses_collection = db["diagnoses"]
papers_collection = db["papers"]
check_process_collection = db["check_process"]
legit_hospitals_collection = db["legit_hospitals"]



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

# Tối ưu hóa hàm upload
async def handle_upload(image: Optional[Upload]) -> Optional[str]:
    if image:
        result = upload_image_bytes(await image.read())
        return result if isinstance(result, str) else result.get("secure_url")
    return None

# Paper logic
async def create_paper(title: str, content: str, image: Upload = None,author: Optional[str] = None, authorImage: Optional[Upload] = None, authorDescription: Optional[str] = None, tags: Optional[List[str]] = None):
    image_url = await handle_upload(image)
    author_image_url = await handle_upload(authorImage) 
    paper_data = Paper_Model(title=title, content=content, mainImage=image_url,author=author,authorImage=author_image_url , authorDescription=authorDescription, tags=tags)
    # Kiểm tra tiêu đề và nội dung có tồn tại hay chưa
    existing_paper= await papers_collection.find_one({"title": title, "content": content})
    if existing_paper:
        raise HTTPException(status_code=400, detail="Bài viết với tiêu đề và nội dung này đã tồn tại.")
    # Chèn dữ liệu vào cơ sở dữ liệu
    result = await papers_collection.insert_one(paper_data.model_dump(by_alias=True, exclude_none=True))
    new_paper = await papers_collection.find_one({"_id": result.inserted_id})
    if new_paper:
        new_paper["_id"] = str(new_paper["_id"])
        return Paper_Model(**new_paper)
    
    return None

async def get_all_papers():
    papers = []
    async for paper in papers_collection.find():
        paper["_id"] = str(paper["_id"])
        papers.append(Paper_Model(**paper))
    return papers

async def get_paper_by_id(paper_id: str):
    paper = await papers_collection.find_one({"_id": ObjectId(paper_id)})
    if paper:
        paper["_id"] = str(paper["_id"])
        return Paper_Model(**paper)
    return None

async def get_papers_by_tag(tag: str):
    papers = []
    async for paper in papers_collection.find({"tags": tag}):
        paper["_id"] = str(paper["_id"])
        papers.append(Paper_Model(**paper))
    return papers

async def update_paper(paper_id: str, title: str = None, content: str = None, image: Upload = None, author: Optional[str] = None, authorImage: Optional[Upload] = None, authorDescription: Optional[str] = None, tags: Optional[List[str]] = None):
    update_data = {k: v for k, v in [("title", title), ("content", content),("author",author),("authorDescription",authorDescription),("tags",tags)] if v is not None}
    image_url = await handle_upload(image)
    author_image_url = await handle_upload(authorImage)
    if author_image_url:
        update_data["authorImage"] = author_image_url
    if image_url:
        update_data["mainImage"] = image_url
    result = await papers_collection.update_one({"_id": ObjectId(paper_id)}, {"$set": update_data})
    return await get_paper_by_id(paper_id) if result.modified_count else None

async def delete_paper(paper_id: str):
    result = await papers_collection.delete_one({"_id": ObjectId(paper_id)})
    return result.deleted_count > 0

# Check process logic
# Helper: Convert _id sang str
def convert_id(doc):
    if doc and "_id" in doc:
        doc["_id"] = str(doc["_id"])
    return doc

async def create_check_process(user_id: str, image: Upload):
    image_url = await handle_upload(image)
    data = CheckProcessModel(userId=user_id, imageUrl=[image_url])
    
    result = await check_process_collection.insert_one(data.model_dump(by_alias=True, exclude_none=True))
    new_item = await check_process_collection.find_one({"_id": result.inserted_id})
    
    return CheckProcessModel(**convert_id(new_item)) if new_item else None

async def track_check_process(user_id: str, image: Upload, check_process_id: str):
    image_url = await handle_upload(image)
    check_process = await check_process_collection.find_one({
        "_id": ObjectId(check_process_id),
        "userId": user_id
    })

    if not check_process:
        raise ValueError("Không tìm thấy quá trình kiểm tra cho người dùng này")

    # Bảo đảm imageUrl là list
    current_images = check_process.get("imageUrl", [])
    current_images.append(image_url)

    await check_process_collection.update_one(
        {"_id": ObjectId(check_process_id)},
        {"$set": {"imageUrl": current_images}}
    )

    return {
        "first_image": current_images[0] if current_images else None,
        "latest_image": current_images[-1] if current_images else None
    }

async def get_check_process_by_user(user_id: str):
    results = []
    async for item in check_process_collection.find({"userId": user_id}):
        results.append(CheckProcessModel(**convert_id(item)))
    return results

async def get_check_process_by_id(check_process_id: str):
    item = await check_process_collection.find_one({"_id": ObjectId(check_process_id)})
    return CheckProcessModel(**convert_id(item)) if item else None

async def delete_check_process(check_process_id: str):
    result = await check_process_collection.delete_one({"_id": ObjectId(check_process_id)})
    return result.deleted_count > 0

# Hospital logic
async def create_legit_hospital(name: str, address: str, phone: Optional[str] = None, img: Optional[Upload] = None, yearEstablished: Optional[int] = None, specialties: List[str] = [], region: Optional[str] = None,hospitalDescription: Optional[str] = None, rate: Optional[float] = None):
    img_url = await handle_upload(img)
    data = LegitHospitalModel(
        name=name,
        address=address,
        phone=phone,
        img=img_url,
        yearEstablished=yearEstablished,
        specialties=specialties,
        region=region,
        hospitalDescription=hospitalDescription,
        rate=rate
    )
    result = await legit_hospitals_collection.insert_one(data.model_dump(by_alias=True, exclude_none=True))
    new_doc = await legit_hospitals_collection.find_one({"_id": result.inserted_id})
    if new_doc:
        new_doc["_id"] = str(new_doc["_id"])
        return LegitHospitalModel(**new_doc)
    return None

async def get_all_legit_hospitals():
    hospitals = []
    async for hospital in legit_hospitals_collection.find():
        hospital["_id"] = str(hospital["_id"])
        hospitals.append(LegitHospitalModel(**hospital))
    return hospitals

async def get_legit_hospital_by_id(hospital_id: str):
    hospital = await legit_hospitals_collection.find_one({"_id": ObjectId(hospital_id)})
    if hospital:
        hospital["_id"] = str(hospital["_id"])
        return LegitHospitalModel(**hospital)
    return None

async def update_legit_hospital(hospital_id: str, name: str = None, address: str = None, phone: str = None, img: Upload = None, yearEstablished: int = None, specialties: list = None, region: str = None, hospitalDescription: Optional[str] = None, rate: Optional[float] = None):
    update_data = {k: v for k, v in [("name", name), ("address", address), ("phone", phone), ("yearEstablished", yearEstablished), ("specialties", specialties), ("region", region),("hospitalDescription",hospitalDescription),("rate",rate )] if v is not None}
    img_url = await handle_upload(img)
    if img_url:
        update_data["img"] = img_url
    result = await legit_hospitals_collection.update_one({"_id": ObjectId(hospital_id)}, {"$set": update_data})
    return await get_legit_hospital_by_id(hospital_id) if result.modified_count else None

async def delete_legit_hospital(hospital_id: str):
    result = await legit_hospitals_collection.delete_one({"_id": ObjectId(hospital_id)})
    return result.deleted_count > 0

async def add_specialty_to_hospital(hospital_id: str, specialty: str):
    result = await legit_hospitals_collection.update_one({"_id": ObjectId(hospital_id)}, {"$addToSet": {"specialties": specialty}})
    return result.modified_count > 0

async def remove_specialty_from_hospital(hospital_id: str, specialty: str):
    result = await legit_hospitals_collection.update_one({"_id": ObjectId(hospital_id)}, {"$pull": {"specialties": specialty}})
    return result.modified_count > 0

async def get_hospitals_by_specialty(specialty: str):
    hospitals = []
    async for hospital in legit_hospitals_collection.find({"specialties": specialty}):
        hospital["_id"] = str(hospital["_id"])
        hospitals.append(LegitHospitalModel(**hospital))
    return hospitals
