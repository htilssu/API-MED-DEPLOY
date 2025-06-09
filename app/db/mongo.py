from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv
load_dotenv()

MONGODB_URI = os.getenv("MONGO_URI" or None)
DATABASE_NAME = os.getenv("DATABASE_NAME" or "mydatabase")



try:
    client = AsyncIOMotorClient(MONGODB_URI)
    db = client[DATABASE_NAME]
except Exception as e:
    print(f"Lỗi kết nối MongoDB: {e}")
    raise

async def ping_db():
    try:
        await db.command("ping")
        print("Kết nối MongoDB thành công!")
    except Exception as e:
        print(f"Lỗi khi ping MongoDB: {e}")
        raise