from starlette.config import Config

config = Config(".env")

DATABASE_URI = config("DATABASE_URI")
DATABASE_PORT = config("DATABASE_PORT", cast=int)
DATABASE_NAME = config("DATABASE_NAME")
LISTEN_IP = config("HOST")
LISTEN_PORT = config("LISTEN_PORT", cast=int)
AUTH_SECRET = config("AUTH_SECRET")
WORKSPACE_MANAGEMENT_IP_PORT = config("WORKSPACE_MANAGEMENT_IP_PORT")