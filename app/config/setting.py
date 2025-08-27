from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class Setting(BaseSettings):
    # Redis Configuration
    REDIS_HOST: str = 'localhost'
    REDIS_PORT: int = 6379

    # MongoDB Configuration
    MONGO_URI: str = 'mongodb://localhost:27017'
    DATABASE_NAME: str = 'app'

    # Server Configuration
    PORT: int = 8080

    # Cloudinary Configuration
    CLOUD_NAME: Optional[str] = None
    API_KEY: Optional[str] = Field(default=None, alias="api_key")
    API_SECRET: Optional[str] = Field(default=None, alias="api_secret")

    # Email Configuration
    EMAIL_HOST: Optional[str] = None
    EMAIL_PORT: Optional[int] = None
    EMAIL_USER: Optional[str] = None
    EMAIL_PASSWORD: Optional[str] = None

    # External API Keys
    API_WEATHER: Optional[str] = Field(default=None, alias="api_weather")
    MAPBOX_KEY: Optional[str] = Field(default=None, alias="mapbox_key")
    HUGGINGFACE_TOKEN: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None

    # Google Cloud Configuration
    GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


setting = Setting()