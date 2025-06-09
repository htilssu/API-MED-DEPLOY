from fastapi import APIRouter, Depends, HTTPException,status
from typing import List
from app.controller.auth_controller import get_all_users, get_user, create_user, login_user,update_user,delete_user,forgot_password,reset_password
from app.models.userModel import CreateUserModel, LoginModel,UserModel

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
async def create_new_user(user_data: CreateUserModel):
    """
    Tạo người dùng mới.
    """
    try:
        return await create_user(user_data.dict())
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    

@router.put("/users/{user_id}", response_model=UserModel)
async def update_user_by_id(user_id: str, update_data: CreateUserModel):
    return await update_user(user_id, update_data=update_data.dict())

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