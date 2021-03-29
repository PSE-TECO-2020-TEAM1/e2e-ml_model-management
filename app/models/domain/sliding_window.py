from pydantic import BaseModel

class SlidingWindow(BaseModel):
    window_size: int
    sliding_step: int
