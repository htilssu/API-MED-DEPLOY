from typing import Optional
from pydantic import BaseModel, Field
from bson import ObjectId

class TagModel(BaseModel):
    id: Optional[str] = Field(default=None, alias="_id")  
    name: str

    model_config = {
        "arbitrary_types_allowed": True,
        "json_encoders": {
            ObjectId: str
        },
        "populate_by_name": True
    }