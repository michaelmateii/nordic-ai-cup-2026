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

---

---

# EXP-D002 — Dataset geometry and temporal analysis

**Status:** COMPLETE

**Date:** 2026-09-17

**Hypothesis:**  
The supplied Helsinki data contains strong repeated-object temporal structure and several classes that become extremely small at camera level 0. Quantifying object scale and inter-frame movement should determine whether tracking and active zoom deserve priority over conventional frame-independent detection.

**Change:**  
Added reproducible dataset-analysis tooling in `drone/scripts/analyze_dataset.py`.

**Validation:**  
Parsed all supplied Helsinki annotations and measured class frequency, box dimensions at L0/L1/L2, and consecutive-frame object motion.

**Hardware:**  
MacBook Air M1

## Results

- Annotated frames: 25
- Total labeled boxes: 259
- Classes: 16
- Physical identities: 16, one per class
- Consecutive-frame motion comparisons: 243
- Median global dx: +1.0 source px/frame
- Median global dy: +65.0 source px/frame
- Median displacement magnitude: 65.12 source px/frame

### Class appearances

| Class | Frames |
|---|---:|
| condor | 11 |
| hangar | 6 |
| helicopter | 19 |
| jammer | 13 |
| jet_plane | 22 |
| large_launcher | 25 |
| large_tower | 19 |
| medium_launcher | 10 |
| medium_plane | 5 |
| mine_roller | 2 |
| small_launcher | 25 |
| small_plane | 9 |
| small_tower | 20 |
| spacecraft | 23 |
| ta-ta | 25 |
| tank | 25 |

### Representative median object sizes

| Class | Source | L0 | L1 | L2 |
|---|---|---|---|---|
| condor | 173x166 | 43x42 | 87x83 | 173x166 |
| helicopter | 116x94 | 29x24 | 58x47 | 116x94 |
| jammer | 33x43 | 8x11 | 17x22 | 33x43 |
| jet_plane | 77x81 | 19x20 | 39x41 | 77x81 |
| medium_launcher | 47x46 | 12x11 | 24x23 | 47x46 |
| small_launcher | 22x30 | 6x8 | 11x15 | 22x30 |
| spacecraft | 44x49 | 11x12 | 22x25 | 44x49 |
| ta-ta | 32x17 | 8x4 | 16x9 | 32x17 |
| tank | 50x47 | 13x12 | 25x24 | 50x47 |

**Interpretation:**  
The dataset is highly temporal rather than consisting of independent examples. Once an object is identified, its next-frame position should often be substantially easier to predict than rediscovering it from scratch.

Level 0 severely undersamples multiple important classes. Small launcher, ta-ta, jammer, spacecraft, tank and several aircraft become approximately single-digit to low-teens pixel structures. Active zoom therefore has real information value rather than merely enlarging already sufficient imagery.

The strong global +Y motion suggests that a simple constant-velocity tracker, potentially supplemented by global image motion estimation, deserves early testing.

**Decision:** KEEP

Prioritize:
1. class/exemplar recognition,
2. temporal memory/tracking,
3. zoom-aware camera policy,
over expensive frame-independent detector tuning.

---

---

# EXP-D003 — Frozen DINOv2 exemplar recognition screen

**Status:** COMPLETE

**Date:** 2026-09-17

**Hypothesis:**  
A strong pretrained visual representation can distinguish the 16 Drone Flyby classes from supplied full-resolution object exemplars without task-specific training. If recognition works when localization is provided, proposal generation + exemplar classification is worth pursuing.

**Change:**  
Used frozen `facebook/dinov2-small` embeddings on ground-truth object crops padded by 20%. Each observation was classified by cosine similarity to leave-one-observation-out class prototypes.

**Validation:**  
259 supplied Helsinki GT crops.

This is a representation separability screen, not an unbiased estimate of final evaluation performance, because repeated frames contain the same physical object identity.

**Hardware:**  
MacBook Air M1 / Apple MPS

## Results

- Samples: 259
- Overall classification accuracy: 0.9730

### Per-class accuracy

| Class | Correct | Accuracy |
|---|---:|---:|
| condor | 10/11 | 0.909 |
| hangar | 5/6 | 0.833 |
| helicopter | 19/19 | 1.000 |
| jammer | 13/13 | 1.000 |
| jet_plane | 21/22 | 0.955 |
| large_launcher | 25/25 | 1.000 |
| large_tower | 18/19 | 0.947 |
| medium_launcher | 9/10 | 0.900 |
| medium_plane | 5/5 | 1.000 |
| mine_roller | 2/2 | 1.000 |
| small_launcher | 25/25 | 1.000 |
| small_plane | 9/9 | 1.000 |
| small_tower | 20/20 | 1.000 |
| spacecraft | 22/23 | 0.957 |
| ta-ta | 25/25 | 1.000 |
| tank | 24/25 | 0.960 |

### Embedding inference latency

- Mean: 19.9 ms
- Median: 15.1 ms
- p95: 23.3 ms
- Max: 1018.4 ms

The maximum is consistent with one-time model/MPS warm-up; steady-state median and p95 are far lower.

### Similarity confidence

- Median top-1 margin: 0.3710
- P10 top-1 margin: 0.2361

**Interpretation:**  
Frozen DINOv2 features separate the supplied Drone classes extremely well when localization is already known. Recognition therefore appears substantially easier than localization/discovery.

The result does not establish cross-scene accuracy because all observations are repeated views of the same 16 physical identities. However, 97.3% leave-one-observation-out accuracy and healthy similarity margins justify using DINOv2 as a low-training class recognizer.

Steady-state embedding latency is low enough to classify several candidate crops per frame, especially if candidates are batched.

The next critical unknown is proposal recall in actual 960x540 camera views.

**Decision:** KEEP

Proceed with pretrained proposal-generation screening, followed by DINO classification and temporal tracking if proposal recall is adequate.

---

# EXP-D004 — YOLO11n generic proposal recall at Level 0

**Status:** COMPLETE

**Date:** 2026-09-17

**Hypothesis:**  
A COCO-pretrained lightweight YOLO model may provide useful class-agnostic proposals even though Drone Flyby classes do not align with COCO labels.

**Change:**  
Ran pretrained `yolo11n.pt` on simulated Level 0 960x540 views and ignored predicted semantic classes. Measured whether any proposal overlapped each GT object.

**Validation:**  
All 259 Helsinki GT boxes.

**Hardware:**  
MacBook Air M1 / Apple MPS

## Results

### Confidence 0.001

- Mean proposals/frame: 253.7
- Median proposals/frame: 294
- Median latency: 19.0 ms
- p95 latency: 241.9 ms
- Proposal recall @ IoU 0.30: 0.0347
- Proposal recall @ IoU 0.50: 0.0193

### Confidence 0.01

- Mean proposals/frame: 35.8
- Median proposals/frame: 35
- Median latency: 18.3 ms
- p95 latency: 25.7 ms
- Proposal recall @ IoU 0.30: 0.0193
- Proposal recall @ IoU 0.50: 0.0154

### Confidence 0.05

- Mean proposals/frame: 4.4
- Median proposals/frame: 4
- Median latency: 18.1 ms
- p95 latency: 18.7 ms
- Proposal recall @ IoU 0.30: 0.0077
- Proposal recall @ IoU 0.50: 0.0077

### Per-class proposal recall at conf=0.001

| Class | IoU@0.30 | IoU@0.50 |
|---|---:|---:|
| condor | 0.000 | 0.000 |
| hangar | 0.833 | 0.833 |
| helicopter | 0.000 | 0.000 |
| jammer | 0.000 | 0.000 |
| jet_plane | 0.000 | 0.000 |
| large_launcher | 0.160 | 0.000 |
| large_tower | 0.000 | 0.000 |
| medium_launcher | 0.000 | 0.000 |
| medium_plane | 0.000 | 0.000 |
| mine_roller | 0.000 | 0.000 |
| small_launcher | 0.000 | 0.000 |
| small_plane | 0.000 | 0.000 |
| small_tower | 0.000 | 0.000 |
| spacecraft | 0.000 | 0.000 |
| ta-ta | 0.000 | 0.000 |
| tank | 0.000 | 0.000 |

**Interpretation:**  
Generic COCO objectness fails almost completely on this domain. Only the hangar is reliably localized; 14/16 classes have zero proposal recall even at an extremely permissive confidence threshold. The failure is therefore structural, not a confidence-threshold tuning problem.

Steady-state inference itself is fast (~19 ms), but useful recall would require hundreds of proposals per frame and is still only 1.9% at IoU 0.50.

**Decision:** DISCARD

Do not spend competition time tuning generic COCO YOLO as the primary proposal mechanism.

---

# EXP-D005 — Motion-compensated temporal proposals

**Status:** RUNNING

**Date:** 2026-09-17

**Hypothesis:**  
Because Drone Flyby is a sequential 3 FPS scene with strong coherent camera motion, aligning consecutive Level 0 frames and examining residual changes may expose targets that generic COCO objectness misses.

**Change:**  
Estimate previous→current global image motion with ORB features + RANSAC affine alignment. Warp the previous Level 0 frame into the current frame, compute absolute residual, threshold it, and use connected components as class-agnostic proposals.

**Validation:**  
Frames 2–25 of Helsinki. Frame 1 is excluded because no previous image exists.

Metrics:
- proposal recall @ IoU 0.30
- proposal recall @ IoU 0.50
- per-class proposal recall
- proposals/frame
- processing latency
- affine RANSAC inlier count

Residual thresholds:
- 12
- 20
- 30
- 45

**Hardware:**  
MacBook Air M1 / CPU OpenCV

## Results

### Residual threshold 12
- Mean proposals/frame: 179.8
- Median proposals/frame: 179.5
- Median latency: 62.1 ms
- p95 latency: 66.0 ms
- Median affine inliers: 598
- Minimum affine inliers: 564
- Proposal recall @ IoU 0.30: 0.0806
- Proposal recall @ IoU 0.50: 0.0282

### Residual threshold 20 — best recall
- Mean proposals/frame: 261.4
- Median proposals/frame: 264.0
- Median latency: 60.9 ms
- p95 latency: 65.8 ms
- Median affine inliers: 598
- Minimum affine inliers: 564
- Proposal recall @ IoU 0.30: 0.1694
- Proposal recall @ IoU 0.50: 0.0847

### Residual threshold 30
- Mean proposals/frame: 217.8
- Median proposals/frame: 192.5
- Median latency: 60.6 ms
- p95 latency: 66.1 ms
- Proposal recall @ IoU 0.30: 0.1008
- Proposal recall @ IoU 0.50: 0.0444

### Residual threshold 45
- Mean proposals/frame: 83.4
- Median proposals/frame: 81.5
- Median latency: 60.5 ms
- p95 latency: 66.1 ms
- Proposal recall @ IoU 0.30: 0.0323
- Proposal recall @ IoU 0.50: 0.0081

### Best per-class recall at threshold 20

| Class | IoU@0.30 | IoU@0.50 |
|---|---:|---:|
| condor | 0.200 | 0.000 |
| hangar | 0.000 | 0.000 |
| helicopter | 0.222 | 0.111 |
| jammer | 0.000 | 0.000 |
| jet_plane | 0.227 | 0.227 |
| large_launcher | 0.667 | 0.417 |
| large_tower | 0.053 | 0.000 |
| medium_launcher | 0.000 | 0.000 |
| medium_plane | 0.000 | 0.000 |
| mine_roller | 0.000 | 0.000 |
| small_launcher | 0.000 | 0.000 |
| small_plane | 0.000 | 0.000 |
| small_tower | 0.211 | 0.158 |
| spacecraft | 0.273 | 0.000 |
| ta-ta | 0.000 | 0.000 |
| tank | 0.167 | 0.042 |

**Interpretation:**  
Global frame registration is reliable, so low proposal recall is not caused by failed alignment. After motion compensation, most Drone targets still do not form clean residual components distinguishable from terrain/rendering changes.

The best configuration requires approximately 261 proposals/frame yet recovers only 16.9% of GT objects at IoU 0.30. Seven classes have zero proposal recall even at the loose IoU threshold.

`large_launcher` is a notable exception with 66.7% recall @0.30, so motion residuals may later be retained as a cheap auxiliary signal for some large targets, but they are not suitable as the primary discovery mechanism.

**Decision:** DISCARD

Do not optimize motion-residual proposals further during the current competition phase.

---

---

# EXP-D006 — DINOv2 recognition versus camera zoom

**Status:** RUNNING

**Date:** 2026-09-17

**Hypothesis:**  
The strong DINOv2 recognition result from EXP-D003 depends heavily on object pixel scale. L1/L2 should substantially outperform L0 for small Drone targets.

**Change:**  
Simulate the effective pixel resolution of camera levels L0, L1 and L2 from the supplied 3840x2160 source frames, then classify padded GT crops using frozen DINOv2-small and leave-one-observation-out prototypes.

**Validation:**  
259 Helsinki annotations at each effective camera scale.

This measures recognition conditional on correct localization; it does not measure discovery.

**Hardware:**  
MacBook Air M1 / Apple MPS

## Results

| Camera level | Accuracy | Median top-1 margin | Median embedding latency |
|---|---:|---:|---:|
| L0 | 0.9653 | 0.3461 | 15.0 ms |
| L1 | 0.9691 | 0.3877 | 15.0 ms |
| L2 | 0.9730 | 0.3710 | 15.1 ms |

### Notable per-class results

At L0:
- hangar: 0.667
- condor: 0.909
- helicopter: 0.947
- jet_plane: 0.955
- large_tower: 0.947
- small_tower: 0.950
- spacecraft: 0.957
- tank: 0.960
- 8 classes: 1.000

At L1 and L2, overall accuracy remains approximately 97%.

A Transformers channel-dimension warning occurred for one very small L0 crop. This indicates that extremely tiny crops require more careful fixed-shape preprocessing before deployment, so the exact L0 result should not be interpreted as an unbiased production estimate.

**Interpretation:**  
Given correct localization, frozen DINOv2-small recognition is surprisingly insensitive to camera scale on the supplied identities. Recognition is therefore not currently the dominant challenge.

The much larger unresolved problem is finding target locations. Generic COCO YOLO proposals and motion-compensated residual proposals both failed badly, whereas class recognition on known crops is already strong.

Zoom may still be valuable for localization and confidence, but current evidence does not justify using L2 simply to improve DINO classification.

**Decision:** KEEP

Retain DINOv2-small as the leading class-recognition method. Shift experiment budget toward task-specific localization.

---