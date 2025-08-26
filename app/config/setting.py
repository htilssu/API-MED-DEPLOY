from pydantic_settings import BaseSettings


class Setting(BaseSettings):
    REDIS_HOST: str = 'localhost'
    REDIS_PORT: int = 6379

    MONGO_URI: str = 'mongodb://localhost:27017'
    DATABASE_NAME: str = 'app'


setting = Setting()