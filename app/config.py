from functools import lru_cache

from pydantic import BaseSettings


class Settings(BaseSettings):
    DATABASE_URI: str
    DATABASE_PORT: int
    DATABASE_NAME: str
    HOST: str
    PORT: int
    AUTH_SECRET: str
    WORKSPACE_MANAGEMENT_IP_PORT: str

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings():
    return Settings()
