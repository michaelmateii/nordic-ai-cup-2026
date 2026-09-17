# Nordic AI Cup 2026 — Challenge 2: Drone Flyby

## Experiment protocol

Every experiment uses an `EXP-Dxxx` identifier.

Record:

- Date/time
- Hypothesis
- Change
- Validation method
- mAP@0.50
- AP@0.50 per class
- Mean / median / p95 latency
- Frames skipped in realtime evaluation
- Hardware
- Interpretation
- Decision: KEEP / DISCARD / INVESTIGATE

---

# EXP-D001 — Official baseline and evaluator sanity check

**Status:** COMPLETE

**Date:** 2026-09-17

**Hypothesis:**  
The official evaluator, API and supplied Helsinki data work correctly on the Mac. The oracle should score exactly 1.000, while the supplied edge-detection baseline should score approximately zero and comfortably keep up with the 3 FPS frame clock.

**Change:**  
No modelling change. Official supplied baseline unchanged.

**Validation:**  
1. Oracle on Helsinki.
2. Official baseline offline on all 25 frames.
3. Official baseline realtime at 3 FPS.

**Hardware:**  
MacBook Air M1

## Oracle

- COCO mAP@0.50: 1.000
- AP@0.50: 1.000 for all 16 classes

## Official baseline — offline

- Frames in scene: 25
- Frames sent: 25
- Frames skipped: 0
- Frames unanswered: 0
- Responses accepted: 25
- Timeouts: 0
- HTTP errors: 0
- Invalid responses: 0
- Camera moves applied: 25
- Camera moves refused: 0
- Round trip latency:
  - mean: 24 ms
  - median: 23 ms
  - max: 36 ms
- COCO mAP@0.50: 0.000
- AP@0.50: 0.000 for all 16 classes

## Official baseline — realtime 3 FPS

- Frames in scene: 25
- Frames sent: 25
- Frames skipped: 0
- Frames unanswered: 0
- Responses accepted: 25
- Timeouts: 0
- HTTP errors: 0
- Invalid responses: 0
- Camera moves applied: 25
- Camera moves refused: 0
- Round trip latency:
  - mean: 23 ms
  - median: 23 ms
  - max: 26 ms
- COCO mAP@0.50: 0.000
- AP@0.50: 0.000 for all 16 classes

**Interpretation:**  
The evaluator and API are functioning correctly. Oracle scoring exactly 1.000 validates the scoring pipeline. The supplied example baseline provides no useful detection performance, as expected.

The realtime baseline processes all 25 frames without skipping any frames. API/evaluator overhead is approximately 23 ms median, which is far below the approximately 333 ms frame interval. This leaves substantial compute budget for a real vision model. Future experiments should target well below 333 ms total response time, preferably below approximately 200–250 ms to retain safety margin.

**Decision:** KEEP

The evaluator/API infrastructure is trustworthy and should be used as the reference harness for all subsequent Drone experiments.