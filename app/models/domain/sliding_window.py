from pydantic import BaseModel

class SlidingWindow(BaseModel):
    window_size: int
    sliding_step: int

    def __eq__(self, other):
        if isinstance(other, SlidingWindow):
            return (self.window_size == other.window_size) and (self.sliding_step == other.sliding_step)
        return False
