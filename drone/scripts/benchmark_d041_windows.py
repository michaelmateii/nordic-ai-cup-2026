import sys
import time
import statistics
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from model_runtime import detect

image = np.zeros((540, 960, 3), dtype=np.uint8)
region = [960, 540, 2880, 1620]

for _ in range(3):
    detect(image, region, 3840, 2160)

times = []
for _ in range(30):
    t0 = time.perf_counter()
    detect(image, region, 3840, 2160)
    times.append((time.perf_counter() - t0) * 1000.0)

times_sorted = sorted(times)
print("mean:", round(statistics.mean(times), 1), "ms")
print("median:", round(statistics.median(times), 1), "ms")
print("p95:", round(times_sorted[int(len(times) * 0.95) - 1], 1), "ms")
print("max:", round(max(times), 1), "ms")
