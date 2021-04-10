from app.fastapi.exception_handling import custom_http_exception_handler, custom_non_existent_error_handler, custom_value_error_handler
from starlette.exceptions import HTTPException
from app.db.asyncdb import initialize_async_db_connection
from app.db.error.non_existent_error import NonExistentError
import uvicorn
from fastapi import FastAPI
from multiprocessing import set_start_method
from app.fastapi.api import router
from app.core.config import LISTEN_IP, LISTEN_PORT
from app.ml.prediction.prediction_manager import prediction_manager

app = FastAPI(title="Model-Management")

app.include_router(router)

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return await custom_http_exception_handler(request, exc)


@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return await custom_value_error_handler(request, exc)

@app.exception_handler(NonExistentError)
async def non_existent_error_handler(request, exc):
    return await custom_non_existent_error_handler(request, exc)


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
