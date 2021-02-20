import uvicorn
from fastapi import FastAPI

from app.config import Settings, get_settings
from app.db import db
from app.routes import router
from app.process_pool import create_executors

app = FastAPI(title="Model-Management")

settings: Settings = get_settings()

app.include_router(router)


@app.on_event("startup")
async def startup():
    db.connect_to_database(settings.client_uri, settings.client_port, settings.db_name)
    create_executors()


@app.on_event("shutdown")
async def shutdown():
    db.disconnect_from_database()

if __name__ == "__main__":
    uvicorn.run(app, host=settings.host, port=settings.port)
