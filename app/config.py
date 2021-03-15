from functools import lru_cache

from pydantic import BaseSettings


class Settings(BaseSettings):
    client_uri: str
    client_port: int
    db_name: str
    host: str
    port: int
    secret_key: str
    workspace_management_ip_port: str

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings():
    return Settings()
