from motor.motor_asyncio import AsyncIOMotorClient

from app.config.setting import setting

try:
    client = AsyncIOMotorClient(setting.MONGO_URI)
    db = client[setting.DATABASE_NAME]
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