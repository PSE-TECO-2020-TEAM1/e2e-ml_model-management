import uvicorn
from fastapi import FastAPI
from multiprocessing import set_start_method

import app.config as config
from app.db import db
from app.routes import router
from app.process_pool import training_executor, prediction_executor

app = FastAPI(title="Model-Management")

app.include_router(router)


@app.on_event("startup")
async def startup():
    db.connect_to_database()
    training_executor.start()
    prediction_executor.start()


@app.on_event("shutdown")
async def shutdown():
    training_executor.shutdown()
    prediction_executor.shutdown()
    db.disconnect_from_database()

if __name__ == "__main__":
    set_start_method("spawn") # Processes are not forked on creation (necessary for FastAPI-Uvicorn)
    uvicorn.run(app, host=config.get_settings().host, port=config.get_settings().port)
