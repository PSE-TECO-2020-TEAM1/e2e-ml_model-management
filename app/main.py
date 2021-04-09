from app.db.asyncdb import initialize_async_db_connection
import uvicorn
from fastapi import FastAPI
from multiprocessing import set_start_method
from app.fastapi.api import router
from app.core.config import LISTEN_IP, LISTEN_PORT
from app.ml.prediction.prediction_manager import prediction_manager

app = FastAPI(title="Model-Management")

app.include_router(router)


@app.on_event("startup")
async def startup():
    initialize_async_db_connection()
    prediction_manager.initiate_clean_up_prediction_process()

@app.on_event("shutdown")
async def shutdown():
    # TODO CLEAN UP PROCESSES IN PREDICTION MANAGER (ADD A METHOD TO PREDICTION MANAGER FOR THE CLEANUP)
    pass

if __name__ == "__main__":
    # Processes are not forked on creation
    # Necessary for FastAPI-Uvicorn: OS signals get inherited and break the server otherwise
    set_start_method("spawn") 
    uvicorn.run(app, host=LISTEN_IP, port=LISTEN_PORT)
