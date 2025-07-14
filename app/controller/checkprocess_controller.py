from app.models.checkprocessModel import CheckProcessModel
from fastapi import APIRouter, UploadFile as Upload, File, HTTPException
from app.db.mongo import db
from bson import ObjectId
from dotenv import load_dotenv
from app.config.cloudinary_config import cloudinary
from io import BytesIO
from typing import List, Optional
from bson.errors import InvalidId

def upload_image_bytes(image_bytes):
    response = cloudinary.uploader.upload(BytesIO(image_bytes))
    return response["secure_url"]

# Tối ưu hóa hàm upload
async def handle_upload(image: Optional[Upload]) -> Optional[str]:
    if image:
        result = upload_image_bytes(await image.read())
        return result if isinstance(result, str) else result.get("secure_url")
    return None

check_process_collection = db["check_process"]


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