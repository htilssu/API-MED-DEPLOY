from fastapi import APIRouter, Request, Response
from app.controller.uv import (check_and_warn_uv)
from app.config.setting import setting

router = APIRouter()


api_key = setting.API_WEATHER

@router.get("/uv-index")
async def uv_index(request: Request, response: Response, lat: float, lon: float):
    """
    Lấy chỉ số UV tại vị trí cụ thể.
    """
    try:
        warning_message,uv_value,level_uv = check_and_warn_uv(lat, lon, api_key)
        return {"message": warning_message, "uv_value": uv_value, "level_uv": level_uv}
    except Exception as e:
        response.status_code = 500
        return {"error": str(e)}