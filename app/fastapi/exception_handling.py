from app.db.error.non_existent_error import NonExistentError
from fastapi.responses import JSONResponse
from fastapi import status
from starlette.exceptions import HTTPException

async def custom_http_exception_handler(request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content=exc.detail)

async def custom_value_error_handler(request, exc: ValueError):
    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=str(exc))

async def custom_non_existent_error_handler(request, exc: NonExistentError):
    return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content=str(exc))