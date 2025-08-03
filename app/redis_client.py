import redis.asyncio as redis
import json

redis_client = redis.Redis(
    # host="10.198.34.44",  # hoặc host của Redis server
    host="localhost",  # Thay đổi thành địa chỉ Redis server của bạn    
    port=6379,
    db=0,
    decode_responses=True  # để tự động decode bytes thành chuỗi
)

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