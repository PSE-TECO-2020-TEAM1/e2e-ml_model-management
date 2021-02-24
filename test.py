import time
from pydantic.fields import Field

from pydantic.main import BaseModel


class A(BaseModel):
    id: str = Field(..., alias="_id")

a = A(_id="1")
print(a.dict())
print(a.dict(by_alias=True, include={"id"}))

start = time.time()

print(time.time() - start)
