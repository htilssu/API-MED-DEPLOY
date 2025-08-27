from app.models.user import UserModel, LoginModel
from app.db.mongo import db
from bson import ObjectId
from datetime import datetime, timedelta
import re
import bcrypt
from fastapi import HTTPException   
import random
import smtplib
from email.message import EmailMessage
from app.redis_client import redis_client, save_result_to_redis,get_result_by_key
from app.config.setting import setting


EMAIL_HOST = setting.EMAIL_HOST
EMAIL_PORT = setting.EMAIL_PORT
EMAIL_USER = setting.EMAIL_USER
EMAIL_PASSWORD = setting.EMAIL_PASSWORD


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
    """
    Cập nhật thông tin người dùng theo ID.
    """
    if not  user_id:
        raise HTTPException(status_code=400, detail="ID người dùng không hợp lệ")

    if not update_data:
        raise HTTPException(status_code=400, detail="Không có dữ liệu để cập nhật")

    result = await users_collection.update_one(
        {"_id": ObjectId(user_id)},
        {"$set": update_data}
    )

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Người dùng không tồn tại")

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

def send_email(to_email: str, subject: str, body: str) -> bool:
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

        print(f"Email sent to {to_email} with subject: {subject}")
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False
    


async def forgot_password(email: str):
    if not email:
        raise HTTPException(status_code=400, detail="Email không được để trống")

    user = await users_collection.find_one({"email": email})
    if not user:
        raise HTTPException(status_code=404, detail="Email không tồn tại")

    # Kiểm tra xem đã gửi gần đây chưa
    resend_flag = await redis_client.get(f"reset_password_sent_{email}")
    if resend_flag:
        raise HTTPException(status_code=429, detail="Vui lòng chờ 1 phút trước khi gửi lại mã xác nhận")

    verification_code = generate_verification_code()

    state_send=send_email(
        to_email=email,
        subject="Mã xác nhận quên mật khẩu",
        body=f"Mã xác nhận của bạn là: {verification_code}. Vui lòng sử dụng mã này để đặt lại mật khẩu. Không chia sẻ mã này với bất kỳ ai."
    )
    if not state_send:
        raise HTTPException(status_code=500, detail="Không thể gửi email. Vui lòng thử lại sau.")

    # Lưu mã xác nhận vào Redis, tồn tại 5 phút
    await redis_client.set(f"reset_password_code_{email}", verification_code, ex=300)

    # Đặt cờ chặn gửi lại trong 1 phút
    await redis_client.set(f"reset_password_sent_{email}", 1, ex=60)

    return {"message": "Mã xác nhận đã được gửi đến email của bạn"}

async def reset_password(email: str, verification_code: str, new_password: str):
    redis_key_code = f"reset_password_code_{email}"
    redis_key_fail = f"reset_password_fail_{email}"

    stored_code = await redis_client.get(redis_key_code)
    if not stored_code:
        raise HTTPException(status_code=400, detail="Mã xác nhận đã hết hạn hoặc không tồn tại")

    # Kiểm tra số lần nhập sai
    fail_count = await redis_client.get(redis_key_fail)
    if fail_count and int(fail_count) >= 5:
        raise HTTPException(status_code=429, detail="Bạn đã nhập sai mã quá nhiều lần. Vui lòng thử lại sau")

    if verification_code != stored_code.decode('utf-8'):
        await redis_client.incr(redis_key_fail)
        await redis_client.expire(redis_key_fail, 300)
        raise HTTPException(status_code=400, detail="Mã xác nhận không đúng")

    # Kiểm tra độ mạnh của mật khẩu mới
    if len(new_password) < 6 or \
       not re.search(r"[!@#$%^&*(),.?\":{}|<>]", new_password) or \
       not re.search(r"[A-Z]", new_password) or \
       not re.search(r"\d", new_password):
        raise HTTPException(status_code=422, detail="Mật khẩu không đủ mạnh")

    user = await users_collection.find_one({"email": email})
    if not user:
        raise HTTPException(status_code=404, detail="Email không tồn tại")

    hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    await users_collection.update_one({"_id": ObjectId(user["_id"])}, {"$set": {"password": hashed_password}})

    # Xoá OTP và fail_count sau khi thành công
    await redis_client.delete(redis_key_code)
    await redis_client.delete(redis_key_fail)

    return {"message": "Mật khẩu đã được đặt lại thành công"}

async def resend_verification_code(email: str):
    # Kiểm tra cooldown resend
    resend_flag = await redis_client.get(f"reset_password_sent_{email}")
    if resend_flag:
        raise HTTPException(status_code=429, detail="Vui lòng đợi ít nhất 1 phút để gửi lại mã xác nhận")

    # Kiểm tra email có tồn tại
    user = await users_collection.find_one({"email": email})
    if not user:
        raise HTTPException(status_code=404, detail="Email không tồn tại")

    verification_code = generate_verification_code()

    send_email(
        to_email=email,
        subject="Mã xác nhận quên mật khẩu (Gửi lại)",
        body=f"Mã xác nhận mới của bạn là: {verification_code}. Không chia sẻ mã này với bất kỳ ai."
    )

    await redis_client.set(f"reset_password_code_{email}", verification_code, ex=300)
    await redis_client.set(f"reset_password_sent_{email}", 1, ex=60)

    return {"message": "Mã xác nhận mới đã được gửi"}   