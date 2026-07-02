"""The one wall-clock budget (design 3.1): every long-running phase checks
the same deadline cooperatively and raises OutOfTime; verify catches it at
the top and reports an honest 'timeout' with whatever was proven so far."""
from __future__ import annotations

import time


class OutOfTime(Exception):
    pass


class Budget:
    def __init__(self, seconds: float, margin: float = 2.0):
        self.t0 = time.time()
        self.deadline = self.t0 + max(0.0, seconds - margin)

    def remaining(self) -> float:
        return self.deadline - time.time()

    def over(self) -> bool:
        return time.time() > self.deadline

    def check(self):
        if self.over():
            raise OutOfTime()


FOREVER = Budget(1e12)
