from dataclasses import dataclass

@dataclass
class SlidingWindow():
    window_size: int
    sliding_step: int
