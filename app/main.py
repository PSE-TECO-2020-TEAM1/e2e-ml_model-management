import uvicorn
from fastapi import FastAPI
from multiprocessing import set_start_method
from app.fastapi.api import router
from app.core.config import LISTEN_IP, LISTEN_PORT

app = FastAPI(title="Model-Management")

app.include_router(router)


@app.on_event("startup")
async def startup():
    print("Starting up...")

@app.on_event("shutdown")
async def shutdown():
    print("Shutting down...")

if __name__ == "__main__":
    # Processes are not forked on creation
    # Necessary for FastAPI-Uvicorn: OS signals get inherited and break the server otherwise
    set_start_method("spawn") 
    uvicorn.run(app, host=LISTEN_IP, port=LISTEN_PORT)
