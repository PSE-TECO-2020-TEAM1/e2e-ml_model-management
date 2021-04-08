from dataclasses import dataclass

@dataclass(frozen=True)
class SlidingWindow():
    window_size: int
    sliding_step: int

    def __str__(self):
        return str(self.window_size) + "_" + str(self.sliding_step)
