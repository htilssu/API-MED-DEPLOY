import redis.asyncio as redis
import json

from app.config.setting import setting

redis_client = redis.Redis.from_url(setting.REDIS_URL)


async def save_result_to_redis(key: str, value: dict, expire: int = 3600) -> bool:
    try:
        result = await redis_client.setex(key, expire, json.dumps(value))
        return result is True  # True nếu lưu thành công
    except Exception as e:
        print(f"Lỗi khi lưu Redis: {e}")
        return False


async def get_result_by_key(key: str):
    value = await redis_client.get(key)  # cần await
    if value is None:
        raise ValueError("Không tìm thấy key trong Redis")
    return json.loads(value)
