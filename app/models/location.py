from typing import Optional
from pydantic import BaseModel, Field

class Location(BaseModel):
    lat: float = Field(..., description="Vĩ độ của vị trí")
    lng: float = Field(..., description="Kinh độ của vị trí")
