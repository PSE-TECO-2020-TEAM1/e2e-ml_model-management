from functools import lru_cache
from pydantic import BaseSettings

class Settings(BaseSettings):
    client_uri: str
    db_name: str
    host: str
    port: int

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()