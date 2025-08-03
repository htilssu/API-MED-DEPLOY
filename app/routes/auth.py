from fastapi import APIRouter, Depends, HTTPException, UploadFile,status
from typing import List
from datetime import date, datetime


from pydantic import EmailStr
from pymssql import Date
from app.controller.auth import get_all_users, get_user, create_user, login_user,update_user,delete_user,forgot_password,reset_password,resend_verification_code
from app.models.user import CreateUserModel, LoginModel,UserModel,UpdateUserModel
from app.config.cloudinary_config import cloudinary
from fastapi import UploadFile as Upload, File, Form
from io import BytesIO
from typing import Optional

def upload_image_bytes(image_bytes):
    response = cloudinary.uploader.upload(BytesIO(image_bytes))
    return response["secure_url"]

async def handle_upload(image: Optional[Upload]) -> Optional[str]:
    if image:
        result = upload_image_bytes(await image.read())
        return result if isinstance(result, str) else result.get("secure_url")
    return None


router = APIRouter()

@router.get("/users", response_model=List[UserModel])
async def get_users():
    """
    Lấy danh sách tất cả người dùng.
    """
    return await get_all_users()

@router.get("/users/{user_id}", response_model=UserModel)
async def get_user_by_id(user_id:str):
    user = await get_user(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User không tồn tại")
    return user

@router.post("/register", response_model=UserModel)
async def create_new_user(
    name: str = Form(...),
    email: EmailStr = Form(...),
    phone: str = Form(...),
    password: str = Form(...),
    dateOfBirth: date = Form(...),
    image: UploadFile = File(None)
):
    try:
        # Chuyển date -> datetime
        dob_datetime = datetime.combine(dateOfBirth, datetime.min.time())

        # Nhớ await handle_upload!
        urlImage = await handle_upload(image)

        user_data = {
            "name": name,
            "email": email,
            "phone": phone,
            "password": password,
            "dateOfBirth": dob_datetime,
            "urlImage": urlImage
        }

        return await create_user(user_data=user_data)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    

@router.put("/users/{user_id}", response_model=UserModel)
async def update_user_by_id(
    user_id: str,
    name: Optional[str] = Form(None),
    email: Optional[str] = Form(None),
    phone: Optional[str] = Form(None),
    image: Optional[Upload] = File(None),
):
    """
    Cập nhật thông tin người dùng theo ID. Hỗ trợ cập nhật ảnh đại diện.
    """
    update_data = {}

    if name: update_data["name"] = name
    if email: update_data["email"] = email
    if phone: update_data["phone"] = phone

    if image:
        url_image = await handle_upload(image)
        update_data["urlImage"] = url_image

    updated_user = await update_user(user_id, update_data)
    return updated_user.model_dump(by_alias=True)

@router.delete("/users/{user_id}")
async def delete_user(user_id: str):
    deleted = await delete_user(user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Không tìm thấy user")
    return {"message": "Xóa user thành công"}

@router.post("/login", response_model=UserModel)
async def login_user_med(data: LoginModel):
    try:
        return await login_user(data)
    except HTTPException as e:
        raise e
    
@router.post("/forgot-password")
async def forgot_password_route(email: str):
    """
    Xử lý quên mật khẩu.
    """
    try:
        return await forgot_password(email)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    
@router.post("/reset-password")
async def reset_password_route(email: str, verification_code: str, new_password: str):
    """
    Đặt lại mật khẩu.
    """
    try:
        return await reset_password(email, verification_code, new_password)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    
@router.post("/resend-verification-code")
async def resend_verification_code_route(email: str):
    """
    Gửi lại mã xác minh.
    """
    try:
        return await resend_verification_code(email)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))