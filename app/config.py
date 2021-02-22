from functools import lru_cache

from pydantic import BaseSettings


class Settings(BaseSettings):
    client_uri: str
    client_port: int
    db_name: str
    host: str
    port: int
    secret_key: str

    SIZE_OF_QUEUE_IN_DATA_WINDOWS = 100000

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings():
    return Settings()
