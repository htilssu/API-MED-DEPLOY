from dotenv import load_dotenv
import os
from fastapi import APIRouter, Request, Response
from app.controller.uv_controller import (check_and_warn_uv)

router = APIRouter()


api_key= os.getenv("api_weather")

@router.get("/uv-index")
async def uv_index(request: Request, response: Response, lat: float, lon: float):
    """
    Lấy chỉ số UV tại vị trí cụ thể.
    """
    try:
        warning_message = check_and_warn_uv(lat, lon, api_key)
        return {"message": warning_message}
    except Exception as e:
        response.status_code = 500
        return {"error": str(e)}