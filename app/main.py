import uvicorn
from fastapi import FastAPI
from app.config import get_settings, Settings
from app.db.mongodb import connect_to_database, disconnect_from_database
from app.routes import router

app = FastAPI(title="Model-Management")

settings: Settings = get_settings()

app.include_router(router)

@app.on_event("startup")
async def startup():
    await connect_to_database(settings.client_uri, settings.db_name)

@app.on_event("shutdown")
async def shutdown():
    await disconnect_from_database()

if __name__ == "__main__":
    uvicorn.run(app, host=settings.host, port=settings.port)