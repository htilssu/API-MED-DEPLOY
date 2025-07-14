from app.models.user import UserModel, LoginModel
from app.db.mongo import db
from bson import ObjectId
from datetime import datetime, timedelta
import re
import bcrypt
from fastapi import HTTPException   
import random
import smtplib
import os
from email.message import EmailMessage
from dotenv import load_dotenv
from app.redis_client import redis_client, save_result_to_redis,get_result_by_key


load_dotenv()

EMAIL_HOST = os.getenv("EMAIL_HOST")
EMAIL_PORT = int(os.getenv("EMAIL_PORT"))
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")


users_collection = db["users"]

async def get_all_users():
    users = []
    async for user in users_collection.find():
        user["_id"] = str(user["_id"])  # Chuyển _id thành str
        users.append(UserModel(**user))
    return users

async def get_user(user_id: str):
    user = await users_collection.find_one({"_id": ObjectId(user_id)})
    if user:
        user["_id"] = str(user["_id"])  # Chuyển _id thành str
        return UserModel(**user)
    return None

async def create_user(user_data: dict):
    name = user_data.get("name")
    if not name:
        raise ValueError("Name is required")
    if len(name) < 5:
        raise ValueError("Name must be at least 5 characters long")
    date_of_birth = user_data.get("dateOfBirth")
    if not date_of_birth:
        raise ValueError("Date of Birth is required")
    # Kiểm tra đủ 18 tuổi (bỏ comment và sửa nếu cần)
    # try:
    #     dob = datetime.strptime(date_of_birth, "%Y-%m-%d")
    #     if (datetime.now() - dob) < timedelta(days=18 * 365):
    #         raise ValueError("You must be at least 18 years old")
    # except ValueError:
    #     raise ValueError("Date of Birth must be in YYYY-MM-DD format")
    password = user_data.get("password")
    if not password:
        raise ValueError("Password is required")
    if len(password) < 6:
        raise ValueError("Password must be at least 6 characters long")
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        raise ValueError("Password must contain at least one special character")
    if not re.search(r"[A-Z]", password):
        raise ValueError("Password must contain at least one uppercase letter")
    if not re.search(r"\d", password):
        raise ValueError("Password must contain at least one number")
    user_data["password"] = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    phone = user_data.get("phone")
    if not phone:
        raise ValueError("Phone is required")
    if not re.match(r"^\d{10,11}$", phone):
        raise ValueError("Phone must be a valid 10 or 11 digit number")
    if await users_collection.find_one({"phone": phone}):  # Thêm await
        raise ValueError("Phone number already exists")
    email = user_data.get("email")
    if not email:
        raise ValueError("Email is required")
    if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        raise ValueError("Email must be a valid email address")
    if await users_collection.find_one({"email": email}):  # Thêm await 
        raise ValueError("Email already exists")

    result = await users_collection.insert_one(user_data)
    return await get_user(str(result.inserted_id))

async def update_user(user_id: str, update_data: dict):
    await users_collection.update_one({"_id": ObjectId(user_id)}, {"$set": update_data})
    return await get_user(user_id)

async def delete_user(user_id: str):
    result = await users_collection.delete_one({"_id": ObjectId(user_id)})
    return result.deleted_count > 0

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

async def login_user(data: LoginModel):
    user = await users_collection.find_one({"email": data.email})
    if not user:
        raise HTTPException(status_code=401, detail="Email không tồn tại")
    if not verify_password(data.password, user["password"]):
        raise HTTPException(status_code=401, detail="Sai mật khẩu")
    
    user["_id"] = str(user["_id"])  # Chuyển _id thành str
    user.pop("password", None)  # Xóa mật khẩu
    return UserModel(**user)

def generate_verification_code():
    return str(random.randint(100000, 999999))

def send_email(to_email: str, subject: str, body: str):
    try:
        msg = EmailMessage()
        msg['From'] = EMAIL_USER
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.set_content(body)

        with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as smtp:
            smtp.starttls()
            smtp.login(EMAIL_USER, EMAIL_PASSWORD)
            smtp.send_message(msg)
        return {"message": "Email sent successfully"}
    except Exception as e:
        return {"error": str(e)}
    


async def forgot_password(email: str):
    if not email:
        raise HTTPException(status_code=400, detail="Email không được để trống")
    user = await users_collection.find_one({"email": email})
    if not user:
        raise HTTPException(status_code=404, detail="Email không tồn tại")
    verification_code = generate_verification_code()
    send_email(
        to_email=email,
        subject="Mã xác nhận quên mật khẩu",
        body=f"Mã xác nhận của bạn là: {verification_code}. Vui lòng sử dụng mã này để đặt lại mật khẩu. Không chia sẻ mã này với bất kỳ ai."
    )
    await redis_client.set(f"reset_password_code_{email}", verification_code, ex=300)  # Lưu mã xác nhận vào Redis với thời gian hết hạn 5 phút

    
    return {"message": "Mã xác nhận đã được gửi đến email của bạn", "verification_code": verification_code}

async def reset_password(email: str, verification_code: str, new_password: str):
    verification_code_redis = await redis_client.get(f"reset_password_code_{email}")
    if not verification_code_redis:
        raise HTTPException(status_code=400, detail="Mã xác nhận đã hết hạn hoặc không hợp lệ")
    
    user = await users_collection.find_one({"email": email})
    if not user:
        raise HTTPException(status_code=404, detail="Email không tồn tại")
    
    if verification_code != verification_code_redis.decode('utf-8'):
        raise HTTPException(status_code=400, detail="Mã xác nhận không hợp lệ")
    
    hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    await users_collection.update_one({"_id": ObjectId(user["_id"])}, {"$set": {"password": hashed_password}})
    
    return {"message": "Mật khẩu đã được đặt lại thành công"}