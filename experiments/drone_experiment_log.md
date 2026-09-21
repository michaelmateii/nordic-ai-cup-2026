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

# EXP-D007 — Task-specific one-class YOLO localization

**Status:** RUNNING

**Date:** 2026-09-17

**Hypothesis:**  
Pooling all 16 Drone Flyby classes into one task-specific `target` category will provide substantially better localization than generic COCO objectness. DINOv2 can handle semantic classification separately.

**Change:**  
Construct Level 0 960x540 YOLO dataset with every labeled Drone object mapped to a single `target` class.

Temporal split:
- train: frames 0–20
- gap: frame 21
- validation: frames 22–24

The split was revised before training because the initial temporal holdout placed `hangar` and `medium_plane` entirely outside the training set. The revised split keeps every validation target morphology represented at least once in training while retaining a one-frame temporal gap.

Expected dataset inventory:
- train: 21 frames / 222 boxes
- validation: 3 frames / 27 boxes

Late-appearing classes remain scarce:
- hangar: 2 training appearances
- medium_plane: 1 training appearance

This remains a same-identity temporal feasibility screen, not a cross-scene generalization estimate.

**Validation:**  
Temporal Helsinki holdout. This contains the same physical identities and therefore measures temporal/viewpoint generalization rather than cross-scene generalization.

Primary metrics:
- box recall
- mAP@0.50
- inference latency
- per-original-class localization recall via separate evaluator after training

**Hardware:**  
MacBook Air M1 / Apple MPS

## Training results

Training stopped by early stopping after 44 epochs.

- Best epoch: 29
- Validation images: 3
- Validation instances: 27
- Precision: 0.983
- Recall: 0.407
- mAP@0.50: 0.435
- mAP@0.50:0.95: 0.228

### M1 validation speed

- Preprocess: 0.6 ms/image
- Inference: 187.6 ms/image
- Postprocess: 7.4 ms/image

Best checkpoint:

`drone/artifacts/exp_d007/runs/yolo11n_objectness/weights/best.pt`

**Interpretation:**  
Task-specific one-class fine-tuning substantially outperforms generic COCO objectness, proving that the supplied labeled sequence contains enough signal to learn Drone-specific localization.

At the default validation operating point, the model is extremely precise (0.983) but recall is only 0.407. Because downstream DINOv2 recognition can reject false proposals, precision is less important than proposal recall for our intended two-stage system.

Inference is approximately 188 ms on the M1, leaving limited but usable room within the ~333 ms frame interval. Deployment latency will need explicit benchmarking with DINO and API overhead.

**Decision:** INVESTIGATE

Before retraining, sweep low confidence thresholds on `best.pt` and measure proposal recall by original Drone class.

---

---

# EXP-D008 — Low-confidence task-specific proposal sweep

**Status:** COMPLETE

**Date:** 2026-09-17

**Hypothesis:**  
EXP-D007's very high precision and moderate recall may reflect an overly conservative confidence operating point. Lower thresholds could make the trained one-class detector useful as a high-recall proposal generator for downstream DINOv2 classification.

**Change:**  
Swept EXP-D007 `best.pt` over confidence thresholds 0.001–0.25.

**Validation:**  
1. All 25 Helsinki frames — diagnostic, heavily in-sample.
2. Temporal holdout frames 22–24 — primary screen.

**Hardware:**  
MacBook Air M1 / Apple MPS

## Results

| Conf | Mean proposals/frame | Median latency | Holdout recall @0.30 | Holdout recall @0.50 |
|---:|---:|---:|---:|---:|
| 0.001 | 300.0 | 19.9 ms | 0.5185 | 0.4815 |
| 0.005 | 300.0 | 17.8 ms | 0.5185 | 0.4815 |
| 0.010 | 79.2 | 17.6 ms | 0.4815 | 0.4815 |
| 0.025 | 20.0 | 17.5 ms | 0.4074 | 0.4074 |
| 0.050 | 11.7 | 17.5 ms | 0.4074 | 0.4074 |
| 0.100 | 8.3 | 17.7 ms | 0.4074 | 0.4074 |
| 0.250 | 6.2 | 17.7 ms | 0.3704 | 0.3704 |

### Diagnostic all-frame result at conf=0.01

- Recall @ IoU 0.30: 0.6988
- Recall @ IoU 0.50: 0.6795

### Holdout per-class recall at most permissive operating point

| Class | IoU@0.30 | IoU@0.50 |
|---|---:|---:|
| hangar | 0.000 | 0.000 |
| jet_plane | 1.000 | 1.000 |
| large_launcher | 1.000 | 1.000 |
| large_tower | 0.667 | 0.333 |
| medium_launcher | 0.000 | 0.000 |
| medium_plane | 1.000 | 1.000 |
| small_launcher | 0.000 | 0.000 |
| spacecraft | 0.000 | 0.000 |
| ta-ta | 0.000 | 0.000 |
| tank | 1.000 | 1.000 |

**Interpretation:**  
Lowering confidence does not solve the main recall problem. Below approximately 0.01 the detector rapidly saturates the 300-detection cap without discovering additional holdout targets.

`conf=0.01` is currently the best practical proposal operating point: it preserves the maximum observed IoU@0.50 holdout recall of 0.4815 while reducing proposal count from 300 to approximately 79/frame.

The failure is highly class-dependent. Several medium/large classes localize reliably, while small launcher, spacecraft, ta-ta and other small/late target morphologies are entirely missed.

This strongly suggests spatial resolution / feature-map scale is now more important than confidence tuning.

Direct warmed inference is ~18 ms on M1, substantially lower than the post-training Ultralytics validation timing.

**Decision:** KEEP

Retain task-specific YOLO as the leading localization baseline. Test spatial tiling before additional training.

---

---

# EXP-D009 — Tiled inference for tiny-object localization

**Status:** RUNNING

**Date:** 2026-09-17

**Hypothesis:**  
EXP-D008 misses are concentrated among tiny target classes. Running the same task-specific detector on overlapping image tiles should increase effective target scale and improve localization recall without retraining.

**Change:**  
Evaluate EXP-D007 `best.pt` at `conf=0.01` with:
- full 960x540 frame
- 2x2 tiles with 20% overlap
- 3x2 tiles with 20% overlap

Every tile is independently resized by YOLO to `imgsz=960`, and detections are mapped back into Level 0 frame coordinates.

**Validation:**  
Primary: temporal holdout frames 22–24.  
Secondary: all 25 Helsinki frames.

Metrics:
- IoU@0.30 proposal recall
- IoU@0.50 proposal recall
- per-original-class recall
- proposals/frame
- total tiled inference latency

**Hardware:**  
MacBook Air M1 / Apple MPS

## Results

| Configuration | Tiles/frame | Proposals/frame | Median latency | Holdout recall @0.30 | Holdout recall @0.50 |
|---|---:|---:|---:|---:|---:|
| Full frame | 1 | 79.2 | 18.6 ms | 0.4815 | 0.4815 |
| 2x2, 20% overlap | 4 | 148.9 | 75.4 ms | 0.2963 | 0.2222 |
| 3x2, 20% overlap | 6 | 155.8 | 154.8 ms | 0.1111 | 0.0370 |

### Holdout per-class IoU@0.50 recall

| Class | Full | 2x2 | 3x2 |
|---|---:|---:|---:|
| hangar | 0.000 | 0.000 | 0.000 |
| jet_plane | 1.000 | 1.000 | 0.333 |
| large_launcher | 1.000 | 1.000 | 0.000 |
| large_tower | 0.333 | 0.000 | 0.000 |
| medium_launcher | 0.000 | 0.000 | 0.000 |
| medium_plane | 1.000 | 0.000 | 0.000 |
| small_launcher | 0.000 | 0.000 | 0.000 |
| spacecraft | 0.000 | 0.000 | 0.000 |
| ta-ta | 0.000 | 0.000 | 0.000 |
| tank | 1.000 | 0.000 | 0.000 |

**Interpretation:**  
Applying spatial tiling only at inference introduces a severe scale/context distribution shift relative to the full-frame training distribution. The existing full-frame detector performs substantially better than either tiled configuration.

This does not establish that tiled localization is inherently ineffective; it establishes that inference-only tiling with a full-frame-trained model is ineffective.

**Decision:** DISCARD

Do not use inference-only tiling with the EXP-D007 model. Run one matched train-on-tiles / infer-on-tiles screen before abandoning the scale hypothesis.

---

---

# EXP-D010 — Matched 2x2 tile training

**Status:** RUNNING

**Date:** 2026-09-17

**Hypothesis:**  
EXP-D009 failed because tiled inference created a scale/context distribution shift relative to full-frame training. Training and inference on the same 2x2 overlapping tile distribution may improve tiny-object localization.

**Change:**  
Build 2x2 Level 0 tiles with 20% overlap from the same temporal split used in EXP-D007.

Train YOLO11n objectness directly on those tiles.

Objects are assigned to tiles containing the object center to avoid duplicate partial labels.

Training:
- YOLO11n pretrained
- imgsz 960
- one class: `target`
- no mosaic
- same temporal train/gap/validation split

**Validation:**  
Matched tiled validation from frames 22–24.

After training, the model will be evaluated after mapping tiled detections back into full-frame Level 0 coordinates.

**Hardware:**  
MacBook Air M1 / Apple MPS

## Results

Matched 2x2 tile training completed successfully.

Validation:
- Images: 12 tiles
- Instances: 27
- Precision: 0.817
- Recall: 0.667
- mAP@0.50: 0.682
- mAP@0.50:0.95: 0.450

Ultralytics validation speed:
- Preprocess: 0.4 ms/tile
- Inference: 51.7 ms/tile
- Postprocess: 2.0 ms/tile

Best checkpoint:

`drone/artifacts/exp_d010/runs/yolo11n_tiled_objectness/weights/best.pt`

**Interpretation:**  
Matched tile training substantially outperforms both full-frame objectness training and inference-only tiling.

EXP-D007 full-frame validation:
- recall: 0.407
- mAP@0.50: 0.435

EXP-D010 matched-tile validation:
- recall: 0.667
- mAP@0.50: 0.682

This confirms that effective target scale is a major localization bottleneck. The failure of EXP-D009 was caused by train/inference scale mismatch rather than tiling itself.

The next step is to map tiled detections back into full Level 0 coordinates and measure original-object proposal recall, proposal count, and end-to-end frame latency.

**Decision:** KEEP

---

---

# EXP-D011 — Full-frame mapped evaluation of tiled detector

**Status:** RUNNING

**Date:** 2026-09-17

**Hypothesis:**  
The matched-tile model from EXP-D010 should substantially improve original-object proposal recall once tile predictions are mapped back into full Level 0 coordinates.

**Change:**  
Run the EXP-D010 model over all four 2x2 overlapping tiles, map detections back into 960x540 Level 0 coordinates, merge duplicate boxes with NMS, and sweep confidence.

**Validation:**  
Primary: frames 22–24 temporal holdout.  
Secondary: all Helsinki frames.

Metrics:
- original-object proposal recall @ IoU 0.30
- original-object proposal recall @ IoU 0.50
- per-class recall
- proposals/frame after tile merging
- total four-tile frame latency

**Hardware:**  
MacBook Air M1 / Apple MPS

## Results

| Confidence | Proposals/frame | Median latency | p95 latency | Holdout recall @0.30 | Holdout recall @0.50 |
|---:|---:|---:|---:|---:|---:|
| 0.005 | 26.1 | 78.8 ms | 82.6 ms | 0.8148 | 0.8148 |
| 0.010 | 16.2 | 76.8 ms | 78.8 ms | 0.7407 | 0.7407 |
| 0.025 | 12.0 | 78.7 ms | 79.5 ms | 0.7037 | 0.7037 |
| 0.050 | 10.9 | 77.8 ms | 88.4 ms | 0.7037 | 0.7037 |
| 0.100 | 10.2 | 75.9 ms | 82.6 ms | 0.6667 | 0.6667 |
| 0.250 | 9.8 | 75.6 ms | 78.3 ms | 0.6667 | 0.6667 |

### Best operating point

`conf=0.005`

- Holdout proposal recall @ IoU 0.50: 0.8148
- Mean proposals/frame: 26.1
- Median total four-tile latency: 78.8 ms

### Per-class holdout proposal recall

| Class | R@0.30 | R@0.50 |
|---|---:|---:|
| hangar | 0.667 | 0.667 |
| jet_plane | 1.000 | 1.000 |
| large_launcher | 1.000 | 1.000 |
| large_tower | 1.000 | 1.000 |
| medium_launcher | 0.500 | 0.500 |
| medium_plane | 0.333 | 0.333 |
| small_launcher | 1.000 | 1.000 |
| spacecraft | 0.000 | 0.000 |
| ta-ta | 1.000 | 1.000 |
| tank | 1.000 | 1.000 |

**Interpretation:**  
Matched tiled training solves much of the tiny-object localization problem. Original-object holdout recall increases to 81.5% at IoU 0.50 while retaining sub-100 ms detector latency on the M1.

Compared with EXP-D008, tiny classes such as `small_launcher` and `ta-ta` improve from zero recall to perfect recall on the temporal holdout.

Remaining localization weaknesses are concentrated in `spacecraft`, `medium_plane`, `medium_launcher`, and to a lesser extent `hangar`.

At this point further detector architecture tuning has lower expected value than integrating the already-strong DINOv2 class recognizer.

**Decision:** KEEP

Use `conf=0.005` as the high-recall proposal operating point for the first complete detection + classification pipeline.

---

---

# EXP-D012 — Tiled YOLO + batched DINO first complete CV pipeline

**Status:** RUNNING

**Date:** 2026-09-17

**Hypothesis:**  
The high-recall tiled YOLO detector from EXP-D011 combined with frozen DINOv2 recognition should produce a meaningful 16-class AP@0.50 score without task-specific semantic detector training.

**Change:**  
Pipeline:
1. 2x2 matched tiled YOLO11n localization at conf=0.005.
2. Map/NMS proposals into Level 0 coordinates.
3. Pad proposal crops by 20%.
4. Embed all frame proposals in one batched DINOv2-small forward pass.
5. Classify by cosine similarity to per-class prototypes built only from training frames 0–20.
6. Score class-specific AP@0.50 on holdout frames 22–24.

**Validation:**  
Temporal holdout frames 22–24.

Same physical identities occur in training and holdout, so this remains a pipeline feasibility estimate rather than cross-scene generalization.

Metrics:
- macro mAP@0.50
- per-class AP@0.50
- proposal count
- YOLO latency
- batched DINO latency
- total per-frame latency

**Hardware:**  
MacBook Air M1 / Apple MPS

## Results

- Macro mAP@0.50: 0.6842
- Mean proposals/frame: 23.0

### Per-class AP@0.50

| Class | AP@0.50 |
|---|---:|
| hangar | 0.0000 |
| jet_plane | 1.0000 |
| large_launcher | 1.0000 |
| large_tower | 1.0000 |
| medium_launcher | 0.5050 |
| medium_plane | 0.3366 |
| small_launcher | 1.0000 |
| spacecraft | 0.0000 |
| ta-ta | 1.0000 |
| tank | 1.0000 |

### Latency

- YOLO median: 89.4 ms
- Batched DINO median: 277.5 ms
- Total median: 389.0 ms
- Total max: 410.6 ms

**Interpretation:**  
The two-stage detector + recognizer architecture is effective: macro AP reaches 0.6842 on the temporal holdout, and classes successfully localized by the tiled detector are generally classified correctly.

The limiting factor is now runtime rather than recognition quality. Batched DINOv2-small adds approximately 278 ms/frame, causing total inference to exceed the approximately 333 ms frame interval.

Per-class AP closely follows localization recall, confirming that improving or accelerating classification is more valuable than further DINO representation work.

**Decision:** KEEP architecture / REPLACE recognizer for realtime deployment.

Retain DINOv2 as a reference recognizer and potential occasional high-confidence fallback. Test a much lighter crop classifier next.

---

# EXP-D013 — Lightweight MobileNetV3 crop recognizer

**Status:** RUNNING

**Date:** 2026-09-17

**Hypothesis:**  
Drone target recognition is sufficiently easy that a lightweight ImageNet-pretrained classifier can replace DINOv2-small with minimal accuracy loss and substantially lower latency.

**Change:**  
Train `mobilenetv3_small_100` as a 16-class crop classifier using Level 0 padded GT crops from frames 0–20.

Validation uses GT crops from frames 22–24.

Primary purpose is recognizer feasibility and speed, not cross-scene accuracy.

**Validation:**  
- crop classification accuracy
- later: classification of actual EXP-D011 predicted boxes
- batched inference latency

**Hardware:**  
MacBook Air M1 / Apple MPS

**Results:**  
TBD

**Interpretation:**  
TBD

**Decision:**  
TBD

---

## Results

- Best epoch: 15
- Best validation crop accuracy: 0.9630
- Model: `mobilenetv3_small_100`
- Training crops: 222
- Temporal holdout crops: 27

Best checkpoint:

`drone/artifacts/exp_d013/classifier/mobilenetv3_small_best.pt`

**Interpretation:**  
A lightweight ImageNet-pretrained MobileNetV3-Small classifier achieves 96.3% accuracy on the temporal GT-crop holdout, very close to the frozen DINOv2 results.

This suggests that Drone Flyby semantic recognition does not require a heavy embedding backbone once localization is provided.

The decisive next test is classification of actual EXP-D011 predicted boxes and full pipeline AP/latency.

**Decision:** KEEP

---

# EXP-D014 — Tiled YOLO + MobileNetV3 realtime pipeline

**Status:** RUNNING

**Date:** 2026-09-17

**Hypothesis:**  
MobileNetV3-Small can replace DINOv2-small with minimal AP loss while reducing recognition latency enough to keep the complete pipeline below the ~333 ms realtime frame interval.

**Change:**  
Same EXP-D012 detector pipeline:
- matched 2x2 tiled YOLO11n
- conf=0.005
- map + NMS proposals
- 30% padded crops

Replace DINOv2 recognition with a batched `mobilenetv3_small_100` 16-class classifier trained in EXP-D013.

**Validation:**  
Temporal holdout frames 22–24.

Metrics:
- macro mAP@0.50
- per-class AP@0.50
- detector latency
- batched classifier latency
- complete frame latency

**Hardware:**  
MacBook Air M1 / Apple MPS

## Results

- Macro mAP@0.50: 0.7141
- Mean proposals/frame: 23.0

### Per-class AP@0.50

| Class | AP@0.50 |
|---|---:|
| hangar | 0.4673 |
| jet_plane | 1.0000 |
| large_launcher | 1.0000 |
| large_tower | 1.0000 |
| medium_launcher | 0.5050 |
| medium_plane | 0.1683 |
| small_launcher | 1.0000 |
| spacecraft | 0.0000 |
| ta-ta | 1.0000 |
| tank | 1.0000 |

### Latency

- YOLO median: 88.5 ms
- MobileNet batch median: 220.0 ms
- Total median: 320.9 ms
- Total max: 455.2 ms

**Interpretation:**  
MobileNetV3-Small preserves and slightly improves semantic scoring relative to DINOv2 while reducing median total latency from 389 ms to approximately 321 ms.

However, runtime remains too close to the 333 ms frame interval, and the 455 ms maximum implies that realtime evaluation may still skip frames.

The unexpectedly high MobileNet MPS latency suggests device-launch/synchronization overhead rather than model complexity may dominate. Before changing architecture, benchmark MobileNet on CPU versus MPS and at smaller input resolutions.

**Decision:** KEEP score pipeline / OPTIMIZE runtime.

---

# EXP-D015 — MobileNet runtime device/resolution sweep

**Status:** RUNNING

**Date:** 2026-09-17

**Hypothesis:**  
MobileNetV3's unexpectedly high MPS latency is dominated by accelerator dispatch overhead. CPU inference and/or reduced crop resolution may preserve AP while substantially reducing runtime.

**Change:**  
Use the exact fixed EXP-D014 YOLO proposals and compare the same MobileNet checkpoint at:

- MPS 224x224
- MPS 160x160
- MPS 128x128
- CPU 224x224
- CPU 160x160
- CPU 128x128

YOLO proposals are generated once and held constant so only classifier behavior changes.

**Validation:**  
Frames 22–24.

Metrics:
- macro mAP@0.50
- classifier median/max latency
- estimated detector + classifier median latency

**Hardware:**  
MacBook Air M1

## Results

| Device / input | Macro mAP@0.50 | Classifier median | Estimated total median |
|---|---:|---:|---:|
| MPS 224 | 0.7141 | 164.6 ms | 249.8 ms |
| MPS 160 | 0.6010 | 163.3 ms | 248.5 ms |
| MPS 128 | 0.5396 | 160.1 ms | 245.2 ms |
| CPU 224 | 0.7141 | 1069.6 ms | 1154.8 ms |
| CPU 160 | 0.6010 | 601.5 ms | 686.7 ms |
| CPU 128 | 0.5396 | 461.1 ms | 546.3 ms |

**Interpretation:**  
Apple MPS is decisively faster than CPU for MobileNetV3 on this workload.

Reducing classifier input resolution from 224 to 160 or 128 provides almost no useful MPS latency reduction while causing substantial AP loss. Therefore 224x224 should remain the recognition input size.

The estimated detector + classifier median at MPS/224 is approximately 250 ms, but EXP-D014 measured ~321 ms end-to-end because preprocessing, crop conversion, tensor construction and other Python overhead are not included in this estimate.

**Decision:** KEEP

Use MobileNetV3-Small on MPS at 224x224. Optimize proposal count and complete end-to-end runtime rather than reducing classifier resolution.

---

# EXP-D016 — End-to-end realtime operating point

**Status:** RUNNING

**Date:** 2026-09-17

**Hypothesis:**  
A slightly higher YOLO confidence threshold will reduce MobileNet batch size enough to improve complete pipeline latency while retaining most of the static AP.

**Change:**  
Evaluate the full matched-tile YOLO + MobileNetV3 pipeline at:

- YOLO conf 0.005
- YOLO conf 0.010
- YOLO conf 0.025

MobileNet remains:
- MPS
- 224x224

Each configuration is warmed before its measured run.

**Validation:**  
Temporal holdout frames 22–24.

Metrics:
- macro mAP@0.50
- mean proposals/frame
- YOLO median latency
- MobileNet median latency
- total median latency
- total maximum latency

**Hardware:**  
MacBook Air M1 / Apple MPS

## Results

| YOLO conf | Macro mAP@0.50 | Mean proposals/frame | YOLO median | MobileNet median | Total median | Total max |
|---:|---:|---:|---:|---:|---:|---:|
| 0.005 | 0.7141 | 23.0 | 87.0 ms | 211.7 ms | 319.9 ms | 376.8 ms |
| 0.010 | 0.6842 | 11.3 | 83.6 ms | 202.2 ms | 295.1 ms | 306.4 ms |
| 0.025 | 0.6505 | 7.7 | 82.9 ms | 161.2 ms | 250.5 ms | 262.8 ms |

### Per-class AP at conf=0.010

| Class | AP@0.50 |
|---|---:|
| hangar | 0.3366 |
| jet_plane | 1.0000 |
| large_launcher | 1.0000 |
| large_tower | 1.0000 |
| medium_launcher | 0.5050 |
| medium_plane | 0.0000 |
| small_launcher | 1.0000 |
| spacecraft | 0.0000 |
| ta-ta | 1.0000 |
| tank | 1.0000 |

**Interpretation:**  
`conf=0.005` provides the highest static score but produces unsafe realtime latency, exceeding the approximately 333 ms frame interval in the measured run.

`conf=0.025` provides substantial latency headroom but sacrifices approximately 0.034 mAP relative to `conf=0.010` and approximately 0.064 relative to `conf=0.005`.

`conf=0.010` is currently the best realtime operating point:
- macro mAP@0.50: 0.6842
- median complete latency: 295.1 ms
- measured maximum latency: 306.4 ms
- approximately 11 proposals/frame

This preserves most of the scoring performance while remaining below the 333 ms frame cadence in the measured holdout run.

**Decision:** KEEP

Use `YOLO_CONF=0.010` as the default realtime static pipeline operating point.

Retain `0.005` as an offline/high-recall option and possible occasional discovery mode if temporal scheduling later creates compute headroom.

---

# EXP-D017 — Oracle-assisted tracking upper-bound screen

**Status:** RUNNING

**Date:** 2026-09-17

**Hypothesis:**  
Drone targets persist across consecutive frames with strong predictable motion. Once a target is acquired, simple constant-velocity propagation may recover a substantial fraction of detector misses.

**Change:**  
Run the matched tiled detector at the realtime operating point (`conf=0.01`) over all 25 Helsinki frames.

For diagnostic purposes only, use GT identity matching to associate successful detections with physical target classes. Propagate acquired boxes forward using constant velocity and combine propagated boxes with fresh detector proposals.

This uses oracle association and is therefore not a deployable tracker; it measures the upper bound/value of temporal persistence.

**Validation:**  
All 25 Helsinki frames.

Metrics:
- detector-only proposal recall @ IoU 0.50
- detector + propagated-track recall @ IoU 0.50
- absolute recall gain
- per-class recall gain
- detector latency

**Hardware:**  
MacBook Air M1 / Apple MPS

## Results

- Total GT instances: 259
- Detector-only hits: 249
- Detector-only recall @ IoU 0.50: 0.9614
- Propagated-track hits: 223
- Combined detector + propagation hits: 250
- Combined recall @ IoU 0.50: 0.9653
- Absolute recall gain: +0.0039
- Median tiled-detector latency: 84.3 ms

### Per-class recall

| Class | Detector | Detector + tracking |
|---|---:|---:|
| condor | 1.000 | 1.000 |
| hangar | 0.500 | 0.667 |
| helicopter | 1.000 | 1.000 |
| jammer | 1.000 | 1.000 |
| jet_plane | 1.000 | 1.000 |
| large_launcher | 1.000 | 1.000 |
| large_tower | 1.000 | 1.000 |
| medium_launcher | 0.900 | 0.900 |
| medium_plane | 0.200 | 0.200 |
| mine_roller | 1.000 | 1.000 |
| small_launcher | 1.000 | 1.000 |
| small_plane | 1.000 | 1.000 |
| small_tower | 1.000 | 1.000 |
| spacecraft | 0.957 | 0.957 |
| ta-ta | 1.000 | 1.000 |
| tank | 0.960 | 0.960 |

**Interpretation:**  
Even with oracle identity association, constant-velocity temporal propagation adds only one additional correctly localized GT instance across the entire Helsinki sequence.

The 96.1% detector-only recall is not a generalization estimate because most frames overlap with detector training. However, the negligible +0.39 percentage-point tracking gain indicates that simple propagation is unlikely to justify substantial engineering effort.

Tracking may still be useful later for identity stabilization or reducing how often classification is required, but it should not currently be treated as a primary score-improvement mechanism.

**Decision:** DISCARD as a score-priority direction.

Do not implement a sophisticated tracker now.

---

# EXP-D018 — Validation-sequence capture infrastructure

**Status:** RUNNING

**Date:** 2026-09-17

**Hypothesis:**  
The officially permitted validation sequence can provide a much larger unlabeled dataset from the real challenge distribution. Capturing the complete L0 view on every received frame maximizes retained visual information for domain analysis, pseudo-labeling and camera-policy development.

**Change:**  
Added competition-safe request capture infrastructure.

For every `/predict` request retain:
- sequence_id
- source frame
- frame_index
- request_id
- timing budgets
- complete transmitted PNG
- resolution level
- camera center
- view ID
- source_region_xyxy
- camera constraints
- camera rejection feedback

Capture failures are isolated from prediction with exception handling.

Initial camera policy holds/resets to:
- level 0
- center (1920, 1080)

This captures the complete source frame at 960x540 every received timestep.

**Validation:**  
First validate locally on the 25-frame Helsinki evaluator before using a competition validation attempt.

Success criteria:
- 25/25 responses accepted
- 0 invalid responses
- 0 timeouts
- 25 PNGs captured
- 25 metadata JSON files captured
- no frame_index gaps

**Hardware:**  
MacBook Air M1

## Results

Local realtime Helsinki validation:

- Frames in scene: 25
- Frames sent: 25
- Frames skipped: 0
- Frames unanswered: 0
- Responses accepted: 25
- Timeouts: 0
- HTTP errors: 0
- Invalid responses: 0
- Camera moves applied: 0
- Camera moves refused: 0
- Round-trip latency:
  - mean: 11 ms
  - median: 10 ms
  - max: 20 ms

Capture inventory:
- PNG frames: 25
- Per-frame metadata JSONs: 25
- frame_index sequence: continuous 0–24

Captured L0 geometry:
- transmitted view: 960x540
- resolution level: 0
- camera center: (1920, 1080)
- source_region_xyxy: [0, 0, 3840, 2160]

**Interpretation:**  
Capture infrastructure is competition-safe under the local realtime evaluator. It introduces negligible overhead, preserves every received frame and its camera metadata, and does not create invalid responses or frame skips.

Holding Level 0 retains the complete source frame on every timestep and is therefore appropriate for the first real validation-sequence capture.

**Decision:** KEEP

Integrate the current realtime CV pipeline while retaining capture.

---

# EXP-D019 — Competition-server integration of realtime static pipeline

**Status:** RUNNING

**Date:** 2026-09-17

**Hypothesis:**  
The EXP-D016 operating point can be deployed through the exact competition DTO/API while retaining capture and remaining below the realtime frame cadence.

**Change:**  
Integrate:
- request capture
- Level 0 full-frame hold
- matched 2x2 tiled YOLO11n
- YOLO confidence 0.01
- NMS
- MobileNetV3-Small 224x224 recognition
- frame-global normalized response boxes

Models are loaded once at server startup.

**Validation:**  
Official Helsinki local evaluator in realtime mode.

Success criteria:
- non-zero mAP
- 25 accepted responses
- 0 invalid responses
- 0 timeouts
- 0 skipped frames if possible
- captured frame_index remains continuous

**Hardware:**  
MacBook Air M1 / Apple MPS

## Results

Official realtime Helsinki evaluator:

- Frames in scene: 25
- Frames sent: 12
- Frames skipped: 13
- Responses accepted: 12
- Timeouts: 0
- HTTP errors: 0
- Invalid responses: 0

Round-trip latency:
- Mean: 515 ms
- Median: 313 ms
- Max: 1267 ms

COCO mAP@0.50: 0.449

Notable AP:
- condor: 0.634
- jammer: 0.614
- small_plane: 0.554
- small_tower: 0.554
- helicopter: 0.525
- spacecraft: 0.525
- jet_plane: 0.505
- mine_roller: 0.505
- medium_plane: 0.000

**Interpretation:**  
The integrated detector/classifier produces meaningful predictions, but realtime latency is unstable enough to skip more than half the sequence.

The local isolated benchmarks underestimated true request-path latency. The extreme 1267 ms spike suggests MPS graph/warmup overhead from changing MobileNet batch shapes is a likely contributor.

The server currently warms MobileNet with only one crop, while real frames contain variable proposal batches.

**Decision:** REVISE

Do not use this configuration for the real validation sequence. Stabilize inference batch shape and lower proposal load first.

---

# EXP-D020 — Fixed-batch realtime runtime stabilization

**Status:** COMPLETE

**Date:** 2026-09-17

**Hypothesis:**  
The frame skipping observed in EXP-D019 was caused largely by unstable MPS execution from variable classifier batch shapes and insufficient startup warmup. A fixed MobileNet batch shape plus a slightly higher YOLO threshold should stabilize realtime latency.

**Change:**  
- YOLO confidence increased from 0.01 to 0.025
- MobileNet classifier padded to fixed batch size 16
- YOLO tiles warmed at startup
- MobileNet fixed batch shape warmed three times before serving requests
- capture retained
- camera held at full Level 0

**Validation:**  
Official Helsinki local evaluator with `--realtime`.

## Results

Attempt statistics:

- Frames in scene: 25
- Frames sent: 25
- Frames skipped: 0
- Frames unanswered: 0
- Responses accepted: 25
- Timeouts: 0
- HTTP errors: 0
- Invalid responses: 0
- Camera moves applied: 0
- Camera moves refused: 0

Round-trip latency:

- Mean: 157 ms
- Median: 145 ms
- Max: 414 ms

COCO mAP@0.50: **0.892**

### Per-class AP@0.50

| Class | AP |
|---|---:|
| helicopter | 1.000 |
| jet_plane | 1.000 |
| large_launcher | 1.000 |
| large_tower | 1.000 |
| mine_roller | 1.000 |
| small_launcher | 1.000 |
| small_plane | 1.000 |
| small_tower | 1.000 |
| ta-ta | 1.000 |
| condor | 1.000 |
| jammer | 1.000 |
| tank | 0.960 |
| spacecraft | 0.950 |
| medium_launcher | 0.871 |
| hangar | 0.281 |
| medium_plane | 0.208 |

**Interpretation:**  
Fixed classifier batching and full startup warmup eliminate the severe realtime instability observed in EXP-D019.

Despite one 414 ms outlier, no frames were skipped. Typical request latency is now comfortably below the 333 ms frame interval.

The static pipeline is highly effective on Helsinki, with remaining errors concentrated primarily in `hangar` and `medium_plane`.

Further broad model changes are not justified before observing the real validation distribution.

**Decision:** KEEP

Use this configuration for the first real validation-sequence capture attempt.

---

# EXP-D021 — First real validation capture and domain reconnaissance

**Status:** COMPLETE

**Date:** 2026-09-17

**Validation attempt UUID:**  
`89eafdd018304b53b58c9f305f78a589`

**Validation sequence ID:**  
`9204f05e8ffe46f995edd8c093823393`

**Configuration:** EXP-D020

- YOLO11n tiled objectness detector
- YOLO confidence: 0.025
- 2x2 matched tiling
- MobileNetV3-Small classifier
- classifier input: 224x224
- fixed classifier batch size: 16
- Apple MPS
- camera held at Level 0 full-frame
- validation capture enabled

## Hypothesis

The Helsinki-trained realtime detector/classifier pipeline would retain meaningful accuracy on the real validation sequence, while the capture infrastructure would provide representative validation-domain imagery for further analysis.

## Competition validation result

- Validation sequence length: 249 frames
- Validation score: **0.0010424119335010423**
- Evaluator errors: none
- Attempt completed successfully

Timing:

- submitted: 2026-09-17 18:45:17 UTC
- started: 2026-09-17 18:45:18 UTC
- finished: 2026-09-17 18:46:42 UTC

## Capture coverage

Captured validation requests:

- PNG frames: **146**
- metadata JSON files: **146**
- total sequence frames: 249
- received coverage: **58.6%**
- missing frames: **103**
- missing fraction: **41.4%**

Frame-index range:

- first received `frame_index`: 0
- last received `frame_index`: 248
- first source frame: 1
- last source frame: 249

All captured requests used:

- resolution level: **0**
- source region: **[0, 0, 3840, 2160]**

Therefore the poor score is not caused by an incorrect camera region or Level-1/Level-2 coordinate conversion.

## Frame-loss pattern

The missing-frame pattern was strongly regular.

After an initial gap from frame index 0 to 4, most of the validation sequence alternated approximately between:

- one frame received
- one frame skipped
- one frame received
- one frame skipped

Most individual gaps were exactly one skipped frame, with only a few two-frame gaps.

This indicates a realtime/deployment problem in addition to the visual-model failure.

The request path was not consistently keeping up with the 333 ms frame interval during the real validation run.

## Validation-domain imagery

A contact sheet was generated from 30 evenly spaced samples across the 146 retained Level-0 validation frames.

The real validation scene differs substantially from the supplied Helsinki scene.

Observed validation environments include:

- highways and road infrastructure
- forest and open terrain
- industrial areas
- harbor / marina regions
- dense urban areas
- residential blocks
- rail and transport infrastructure

The visual/background distribution is therefore substantially broader and different from the Helsinki training reference.

## Validation-domain detector proposal analysis

The EXP-D010 tiled YOLO detector was evaluated offline on all 146 captured validation frames using:

- YOLO confidence: 0.025
- same 2x2 tiled inference
- same NMS configuration
- Apple MPS

Results:

- Mean proposals/frame: **4.5**
- Median proposals/frame: **3.0**
- P95 proposals/frame: **13.0**
- Maximum proposals/frame: **18**

Proposal-count thresholds:

- Frames >16 proposals: **2 / 146**
- Frames >32 proposals: **0 / 146**
- Frames >48 proposals: **0 / 146**

YOLO latency:

- Median: **73.5 ms**
- P95: **89.2 ms**

## Interpretation

The validation failure has two separate components.

### 1. Severe visual-domain / instance overfitting

The detector does not suffer from proposal explosion on the validation sequence.

Instead, it produces very few proposals:

- median only 3 detections/frame
- mean only 4.5 detections/frame

This shows that the Helsinki-trained objectness detector largely fails to activate on the real validation domain.

The extremely high Helsinki result:

- realtime Helsinki mAP@0.50: **0.892**

therefore did not represent class-level generalization.

The supplied Helsinki sequence contains repeated observations of the same small set of physical objects, so both the detector and classifier learned Helsinki-specific appearance and identity cues.

The MobileNet validation-crop accuracy and Helsinki temporal holdouts were therefore overly optimistic measures of real-world generalization.

### 2. Realtime / deployment frame loss

Only 146 of 249 validation frames reached the endpoint.

The approximately alternating receive/skip pattern indicates that the real request path also failed to maintain the 3 FPS cadence consistently.

However, frame loss alone cannot explain a validation score of approximately 0.001.

Even with 41.4% of frames missing, a detector that generalized meaningfully on the remaining frames would be expected to score substantially higher.

Therefore the dominant problem is visual generalization, with deployment latency as a secondary problem that must also be fixed.

## Key conclusion

Further optimization against the 25 Helsinki frames is no longer justified.

The competition strategy must pivot from:

`Helsinki-specific supervised detector training`

to:

`validation-domain data acquisition + generalizable recognition/localization`

The retained validation sequence is now more valuable than additional Helsinki tuning.

Level 0 has already discarded substantial small-object detail, so the next repeatable validation attempt should prioritize high-resolution validation-domain data collection using Level 1 and/or Level 2 camera views.

## Decision

**MAJOR PIVOT — KEEP the validation capture infrastructure, DISCARD Helsinki score as a meaningful generalization benchmark.**

Do not:

- spend more time tuning Helsinki-only YOLO
- build a sophisticated tracker
- optimize MobileNet further
- interpret Helsinki mAP as expected competition performance
- run Evaluation

Next priority:

**EXP-D022 — capture-only high-resolution validation scan using Level 1 camera views.**

Objectives:

1. remove model inference from the request path
2. determine whether the remaining frame loss is caused by ngrok/networking
3. capture real validation-domain imagery at substantially higher spatial resolution
4. use retained validation imagery for domain-specific discovery, pseudo-labeling, template/feature matching, and model adaptation
5. only then run another scoring validation

---

# EXP-D022 — Capture-only Level-1 validation scanner

**Status:** COMPLETE / READY FOR REAL VALIDATION CAPTURE

**Date:** 2026-09-17

## Hypothesis

The repeatable validation sequence can be used as legal high-resolution domain-data acquisition.

Removing detector/classifier inference should eliminate realtime frame loss, while systematic Level-1 camera movement provides substantially more spatial detail than Level 0.

## Change

Replaced the scoring predictor with a capture-only camera scanner.

Pipeline:

- persist every received image + request metadata
- return zero annotations
- request Level 1 views
- cycle among four Level-1 target regions
- obey dynamic camera bounds
- obey maximum center movement
- reserve a 1 px movement safety margin

No YOLO or MobileNet inference is performed.

## Local realtime validation

Official Helsinki evaluator with `--realtime`:

- Frames in scene: 25
- Frames sent: 25
- Frames skipped: 0
- Frames unanswered: 0
- Responses accepted: 25
- Timeouts: 0
- HTTP errors: 0
- Invalid responses: 0
- Camera moves applied: **25**
- Camera moves refused: **0**

Round-trip latency:

- Mean: **11 ms**
- Median: **11 ms**
- Maximum: **15 ms**

mAP@0.50: 0.000, intentionally, because the acquisition server returns no detections.

## Interpretation

The capture-only Level-1 scanner is realtime-safe locally.

Compared with the first real scoring validation, which retained only 146/249 frames, this configuration removes model inference from the critical request path and reduces local latency from approximately 145 ms median to approximately 11 ms.

All requested camera movements are now legal after adding a 1 px safety margin to the movement limit.

The next validation attempt should be treated purely as a data-acquisition run. Its score is expected to be zero.

## Decision

**KEEP — USE FOR REAL VALIDATION DATA ACQUISITION**

Next:

- run one real 249-frame validation attempt
- retain all received L1 views and metadata
- measure capture completeness
- analyze actual validation targets at higher spatial resolution
- do not run Evaluation

---

---

# EXP-D023 — Real-validation L1 proposal visual audit

**Status:** COMPLETE

**Date:** 2026-09-17

## Hypothesis

Although the Helsinki-trained objectness detector has low activation on the real validation domain, its L1 proposals might still contain a useful fraction of true challenge targets.

If true, localization could be retained while recognition/domain adaptation was rebuilt.

If false, the Helsinki objectness detector should be discarded as the primary localization mechanism.

## Data

Validation acquisition sequence:

`3224a582bfbf4273a028497662b7aa7c`

- Captured requests: 196 / 249
- Coverage: 78.7%
- L1 views: 194
- L0 views: 2
- Sequence coverage: frame_index 0–248

Major L1 camera positions:

- (1920,1080): 41
- (960,540): 37
- (960,1620): 29
- (2880,1620): 26
- (2880,540): 24

## Change

Ran the EXP-D010 matched tiled YOLO objectness detector on all retained L1 validation views.

Configuration:

- YOLO confidence: 0.025
- 2x2 matched tiling
- NMS IoU: 0.50
- maximum 20 retained proposals/frame
- context-expanded proposal crops
- ranked visual montage

## Results

- L1 frames audited: **194**
- Proposal crops produced: **673**
- Mean retained proposals/L1 frame: **3.47**

Visual inspection of the ranked proposal montage shows that the detector fires predominantly on background structures and textures, including:

- vegetation and forest patches
- roads and road markings
- roofs and buildings
- fields / terrain boundaries
- water / shoreline structure
- industrial and urban texture
- other compact high-contrast background features

Only a small minority of proposals appear visually object-like, and there is not enough evidence that these correspond consistently to the sixteen target classes.

## Interpretation

The Helsinki-trained detector does not provide a sufficiently reliable localization prior on the real validation domain.

The validation failure is therefore not simply:

`good boxes + bad classification`

It is primarily:

`poor validation-domain localization + poor class generalization`

The detector has learned Helsinki-specific appearance and background correlations rather than a generic concept of challenge-object objectness.

Using DINO or MobileNet to classify all 673 YOLO proposals would spend substantial compute on mostly irrelevant background crops and would still cap recall at the detector's poor localization recall.

The EXP-D010 YOLO detector should therefore not be used as the foundation of the next system.

It may remain useful later as one weak proposal source in an ensemble, but not as the primary candidate generator.

## Decision

**DISCARD as primary localization mechanism.**

Do not:

- further tune YOLO thresholds on this detector
- retrain the classifier around these proposals
- build tracking around these proposals
- optimize the existing Helsinki YOLO architecture further

Next priority:

**EXP-D024 — validation-domain temporal object discovery.**

Use the retained sequential L1 imagery itself to discover persistent or repeated compact objects without relying on the Helsinki-trained detector.
TBD

---

# EXP-D024 — Validation-domain temporal median object discovery

**Status:** COMPLETE

**Date:** 2026-09-17

## Hypothesis

Repeated validation L1 views at the same camera center might permit a temporal median background model. Compact deviations from that median could provide object candidates without relying on the Helsinki-trained detector.

## Data

Validation acquisition sequence:

`3224a582bfbf4273a028497662b7aa7c`

Exact view used:

- Resolution level: L1
- Camera center: (1920, 1080)
- Matching frames: 41
- Frame-index span: 26–248

Frame indices:

26, 29, 31, 42, 46, 50, 54, 55, 62, 63, 67, 71, 78, 81, 82, 90, 97, 98, 105, 107, 113, 137, 141, 147, 149, 156, 160, 176, 192, 193, 197, 201, 205, 220, 224, 225, 229, 232, 240, 244, 248

## Change

Constructed a grayscale temporal median from all 41 exact-center L1 views.

For every frame:

- absolute difference from median background
- thresholding
- morphological opening
- dilation
- connected-component candidate extraction
- area filtering

Thresholds tested:

- 20
- 30
- 40

## Results

Threshold 20:

- mean candidates/frame: **142.5**
- median: **124**
- maximum: **319**

Threshold 30:

- mean candidates/frame: **281.7**
- median: **288**
- maximum: **478**

Threshold 40:

- mean candidates/frame: **396.7**
- median: **406**
- maximum: **609**

Visual inspection shows residuals overwhelmingly following:

- vegetation
- tree boundaries
- roads
- buildings
- roofs
- construction areas
- terrain texture
- urban edges

Increasing the threshold fragments the residual mask into more disconnected components, causing candidate counts to increase rather than producing a cleaner target set.

## Interpretation

The underlying assumption was invalid.

A fixed camera center is defined relative to each source frame, not to a fixed geographic/world coordinate.

As the drone progresses through the sequence, L1 (1920,1080) observes different terrain. The 41 selected frames therefore cannot be combined directly into a stationary temporal background model.

The temporal median represents a mixture of unrelated geographic content, so subtraction highlights general scene differences rather than challenge objects.

This method does not provide usable object proposals.

## Decision

**DISCARD**

Do not continue tuning residual thresholds or morphology for an unregistered long-range temporal median.

A future temporal method would require reliable pairwise image registration between nearby overlapping observations first.

---

# EXP-D025 — Direct labeled-template matching on validation L1

**Status:** COMPLETE

**Date:** 2026-09-17

## Hypothesis

If validation reused the same or nearly identical rendered target assets as Helsinki, direct multi-scale normalized template matching might locate targets without learning a detector.

## Data

- Helsinki labeled source-frame crops
- one median-sized template per class
- source-to-L1 nominal scale: 0.5x
- validation sequence: `3224a582bfbf4273a028497662b7aa7c`
- L1 frames scanned: 194
- relative template scales: 0.75, 0.90, 1.00, 1.10, 1.25

## Results

| Class | Best NCC | Top-3 mean |
|---|---:|---:|
| condor | 0.6711 | 0.6694 |
| hangar | 0.6944 | 0.6924 |
| helicopter | 0.4284 | 0.4279 |
| jammer | 0.8170 | 0.7976 |
| jet_plane | 0.7674 | 0.7575 |
| large_launcher | 0.4883 | 0.4793 |
| large_tower | 0.4957 | 0.4928 |
| medium_launcher | 0.5611 | 0.5469 |
| medium_plane | 0.6156 | 0.6131 |
| mine_roller | 0.6539 | 0.6536 |
| small_launcher | 0.7768 | 0.7718 |
| small_plane | 0.6977 | 0.6968 |
| small_tower | 0.6708 | 0.6694 |
| spacecraft | 0.5253 | 0.5217 |
| ta-ta | 0.8231 | 0.8208 |
| tank | 0.7965 | 0.7954 |

## Interpretation

Raw normalized cross-correlation does not provide reliable class-specific localization in the validation domain.

The visually strongest matches are predominantly background features rather than recognizable target instances.

The apparently high scores for several small classes are especially unreliable because very small templates are searched over a very large number of positions and frames, creating strong chance correlations.

Large-object classes provide a clearer negative control: helicopter, large launcher, large tower and spacecraft have weak scores and incorrect visual matches.

There is therefore no evidence that validation objects can be robustly localized by direct pixel-template reuse from Helsinki.

## Decision

**DISCARD as primary localization method.**

Do not spend time tuning NCC thresholds or adding many more raw templates.

Template similarity may remain useful only as a weak secondary feature after a better candidate generator exists.

---

# EXP-D026A — Dense DINO validation localization screen

**Status:** COMPLETE

**Date:** 2026-09-17

## Hypothesis

EXP-D003 showed that DINOv2 embeddings strongly separate the sixteen classes when localization is already known.

The existing pipeline might therefore fail primarily because of proposal generation rather than because the DINO representation itself is unsuitable.

A dense DINO feature map could potentially localize target-like regions directly without YOLO.

## Change

Built normalized DINOv2-Small class prototypes from all labeled Helsinki crops.

For 16 representative real validation L1 frames:

- resized full 960×540 view to 896×504
- performed one DINOv2 forward pass
- retained dense patch-token feature map
- pooled regions according to expected class-specific L1 dimensions
- computed cosine similarity to each Helsinki class prototype
- retained top spatial locations per class
- visually inspected the best three per class

No YOLO proposals and no model training were used.

## Results

| Class | Best | Top-3 | Top-5 |
|---|---:|---:|---:|
| condor | 0.4857 | 0.4853 | 0.4845 |
| hangar | 0.4099 | 0.4087 | 0.4060 |
| helicopter | 0.4262 | 0.4252 | 0.3948 |
| jammer | 0.3515 | 0.3484 | 0.3332 |
| jet_plane | 0.4907 | 0.4884 | 0.4847 |
| large_launcher | 0.4625 | 0.4606 | 0.4579 |
| large_tower | 0.5554 | 0.5477 | 0.5387 |
| medium_launcher | 0.4459 | 0.4426 | 0.4405 |
| medium_plane | 0.4245 | 0.4123 | 0.4044 |
| mine_roller | 0.3790 | 0.3780 | 0.3534 |
| small_launcher | 0.4191 | 0.4160 | 0.3953 |
| small_plane | 0.4461 | 0.4368 | 0.4276 |
| small_tower | 0.4462 | 0.4438 | 0.4424 |
| spacecraft | 0.3784 | 0.3728 | 0.3666 |
| ta-ta | 0.3073 | 0.3048 | 0.3022 |
| tank | 0.3566 | 0.3526 | 0.3504 |

## Visual audit

Top DINO locations are overwhelmingly background:

- grass
- forest
- roads
- roofs
- generic building structure
- terrain texture

There is no consistent visual evidence that the top locations correspond to the intended challenge classes.

The similarities are also relatively low and tightly grouped, with little separation between the best and subsequent matches.

## Interpretation

DINOv2 is effective at distinguishing repeated Helsinki target appearances when target localization is already known, but Helsinki class prototypes do not transfer reliably enough to directly localize the same semantic classes in the validation domain.

This confirms that the domain/instance gap is too large for straightforward feature-space transfer.

Together with EXP-D023 and EXP-D025, there is now strong evidence that further automatic Helsinki-to-validation transfer is low value.

## Decision

**DISCARD as primary localization strategy.**

Stop spending competition time on:

- Helsinki-trained proposal models
- Helsinki raw templates
- Helsinki DINO prototype localization

Next direction:

**EXP-D027 — manual validation-domain seeds + within-validation propagation.**

Instead of transferring class appearance from Helsinki, manually identify a small number of high-confidence validation targets and use the validation sequence itself for same-instance propagation and pseudo-label generation.

---

# EXP-D027A — Manual real-validation seed annotation

**Status:** COMPLETE

**Date:** 2026-09-17

## Hypothesis

A small number of high-confidence real-validation annotations may provide substantially stronger anchors than attempting to transfer object appearance directly from Helsinki.

## Data

Source:

`3224a582bfbf4273a028497662b7aa7c`

A subset of 30 retained L1 validation frames was manually inspected against the sixteen-class Helsinki reference sheet.

Only visually high-confidence targets were annotated.

## Results

- Manual annotations: **23**
- Frames labeled: **18**

Class distribution:

- helicopter: 2
- large_launcher: 2
- large_tower: 6
- medium_plane: 2
- mine_roller: 2
- medium_launcher: 1
- tank: 8

Seven of sixteen classes received at least one real-domain seed.

Six classes have at least two seeds and can therefore be used for direct propagation validation.

## Interpretation

Manual inspection can identify several target classes reliably in the real validation domain.

These real-domain boxes eliminate the cross-domain appearance problem encountered in EXP-D023–D026A.

The next step is not to train immediately, but to quantitatively test whether a seed from one validation frame can recover the corresponding target in another manually labeled validation frame.

## Decision

**KEEP**

Proceed to EXP-D027B leave-one-seed-out propagation benchmark.

---

# EXP-D027B — Manual-seed temporal propagation benchmark

**Status:** COMPLETE

**Date:** 2026-09-17

## Hypothesis

Real-validation manual seeds should be substantially easier to propagate to nearby validation frames than Helsinki-derived templates because both source and target come from the same domain and sequence.

## Data

Manual validation seeds from EXP-D027A:

- total annotations: 23
- labeled frames: 18

Corrected class distribution:

- helicopter: 2
- large_launcher: 2
- large_tower: 6
- medium_launcher: 1
- medium_plane: 2
- mine_roller: 2
- tank: 8

Classes with at least two seeds were evaluated pairwise.

## Change

For every ordered pair of manually labeled boxes from the same class:

- use one seed crop as a template
- search the other seed frame
- multi-scale normalized cross-correlation
- scales: 0.70–1.30
- compare predicted box with manual target box
- success criterion: IoU >= 0.50

Results were stratified by temporal distance.

## Results

### Temporal propagation

| Maximum frame gap | Pairs | R@IoU0.50 | Median IoU | Median NCC |
|---|---:|---:|---:|---:|
| 15 | 12 | **0.667** | **0.686** | **0.966** |
| 30 | 20 | **0.550** | **0.511** | **0.943** |
| 60 | 42 | 0.262 | 0.000 | 0.850 |
| all | 94 | 0.128 | 0.000 | 0.836 |

### All-pair class results

| Class | Pairs | R@IoU0.50 | Median IoU |
|---|---:|---:|---:|
| helicopter | 2 | 0.000 | 0.000 |
| large_launcher | 2 | 0.000 | 0.000 |
| large_tower | 30 | 0.100 | 0.000 |
| medium_plane | 2 | 0.000 | 0.218 |
| mine_roller | 2 | 0.000 | 0.147 |
| tank | 56 | 0.161 | 0.000 |

## Interpretation

Global template re-identification across the full validation sequence is unreliable.

However, propagation over short temporal distances is substantially stronger:

- gap <= 15: 66.7% success at IoU >= 0.50
- median IoU 0.686
- median NCC 0.966

Performance degrades rapidly as temporal distance increases.

This indicates that the useful structure is not class-level template matching but **local same-instance propagation through nearby validation frames**.

The poor all-pair per-class numbers are therefore not a reason to discard propagation; they are primarily caused by attempting to match the same class across distant views where appearance, location and potentially physical instance differ substantially.

## Decision

**KEEP — LOCAL PROPAGATION ONLY**

Do not perform unrestricted full-sequence matching.

Next step:

EXP-D027C — calibrate propagation confidence using manually labeled pairs, then pseudo-label only high-confidence matches within a short temporal window.

---

# EXP-D027C — Local propagation confidence calibration

**Status:** COMPLETE

**Date:** 2026-09-17

## Hypothesis

Propagation confidence can be calibrated from manually labeled validation seed pairs so that pseudo-label generation favors precision rather than recall.

## Data

Source:

`drone/artifacts/exp_d027b/propagation_benchmark.json`

Propagation pairs were stratified by:

- temporal gap
- normalized cross-correlation confidence
- IoU against manual validation boxes

Success criterion:

- IoU >= 0.50

## Results

### Gap <= 10

| NCC threshold | N | P@IoU0.50 | Median IoU |
|---|---:|---:|---:|
| 0.80 | 10 | 0.800 | 0.723 |
| 0.85 | 10 | 0.800 | 0.723 |
| 0.90 | 8 | **0.875** | **0.737** |
| 0.92 | 7 | 0.857 | 0.754 |
| 0.94 | 7 | 0.857 | 0.754 |
| 0.95 | 7 | 0.857 | 0.754 |
| 0.96 | 7 | 0.857 | 0.754 |
| 0.97 | 3 | 0.667 | 0.633 |
| 0.98 | 1 | 1.000 | 0.633 |

### Gap <= 15

Identical to gap <= 10 for the available manually labeled pairs:

- NCC >= 0.90: 8 pairs
- P@IoU0.50: **0.875**
- Median IoU: **0.737**

### Gap <= 20

At NCC >= 0.90:

- N: 12
- P@IoU0.50: 0.750
- Median IoU: 0.642

### Gap <= 30

At NCC >= 0.90:

- N: 12
- P@IoU0.50: 0.750
- Median IoU: 0.642

## Per-class local calibration

### tank

At gap <= 15:

- NCC >= 0.90: 6 / 6 correct
- P@IoU0.50: **1.000**
- Median IoU: **0.754**

The same six examples remain correct through NCC >= 0.96.

### large_tower

At gap <= 15:

- NCC >= 0.90: 1 / 1 correct
- IoU: 0.721

Sample size is too small for a strong class-level conclusion.

### medium_plane

At gap <= 15:

- NCC >= 0.90–0.97: 1 example
- IoU: 0.436
- propagation failure despite high NCC

This demonstrates that NCC confidence alone is not sufficient for every class.

## Interpretation

The useful operating region is a **short temporal window**, not an extremely high NCC threshold.

For the available validation seeds:

- gap <= 10–15 is materially safer than gap 20–30
- NCC >= 0.90 provides the best observed precision/sample tradeoff
- increasing NCC above ~0.96 removes many examples without improving reliability
- reliability differs strongly by class

Therefore pseudo-label generation should use:

1. short temporal distance
2. NCC >= 0.90
3. forward/backward cycle consistency
4. class-specific trust
5. manual audit before training

## Decision

**KEEP**

Proceed to EXP-D027D.

Initial trusted classes:

- tank
- large_tower

Other seeded classes remain audit-only until more evidence exists.

---

# EXP-D027D — Cycle-consistent validation pseudo-label propagation

**Status:** COMPLETE

**Date:** 2026-09-17

## Hypothesis

High-confidence manual validation seeds can be expanded into additional real-domain labels using short-range same-center template propagation, provided that candidates satisfy forward/backward cycle consistency.

## Configuration

Propagation parameters:

- maximum temporal gap: 10 frames
- forward NCC threshold: 0.90
- scale search: 0.80, 0.90, 1.00, 1.10, 1.20
- exact L1 camera-center match required
- backward consistency required
- minimum cycle IoU: 0.50

Initial trusted classes:

- tank
- large_tower

Other classes were retained as audit-only.

## Results

- Raw forward matches: **58**
- Cycle-consistent matches: **57**
- Deduplicated pseudo-labels: **54**
- Trusted pseudo-labels: **37**
- Audit-only pseudo-labels: **17**

Per class:

| Class | Pseudo-labels | Status |
|---|---:|---|
| helicopter | 7 | AUDIT |
| large_launcher | 2 | AUDIT |
| large_tower | 12 | TRUST |
| medium_launcher | 2 | AUDIT |
| medium_plane | 2 | AUDIT |
| mine_roller | 4 | AUDIT |
| tank | 25 | TRUST |

## Visual audit

The cycle-consistent montage is substantially cleaner than previous automated localization experiments.

Trusted tank and large-tower matches generally correspond to coherent repeated target-like objects rather than generic background texture.

Several audit-only helicopter matches are also visually strong and appear to correspond to the intended object.

This is qualitatively different from EXP-D023–D026A, where automated matches were dominated by unrelated background structure.

## Interpretation

Within-validation same-instance propagation is viable.

Cycle consistency removes most weak one-way matches while preserving many strong nearby matches.

The remaining uncertainty is now primarily semantic/class-specific rather than localization failure.

The correct next step is a lightweight human audit of the 54 propagated labels before using them as training data.

## Decision

**KEEP**

Proceed to EXP-D027E:

- manually accept/reject propagated labels
- merge accepted labels with manual seeds
- create a clean validation-domain training set
- only then train a real-domain detector

---

# EXP-D027E — Human audit of cycle-consistent pseudo-labels

**Status:** COMPLETE

**Date:** 2026-09-17

## Hypothesis

Cycle-consistent short-range propagation from manually labeled validation seeds produces pseudo-labels clean enough to use as real-domain supervision after lightweight human verification.

## Input

EXP-D027D generated:

- raw forward matches: 58
- cycle-consistent matches: 57
- deduplicated pseudo-labels: 54

Pseudo-label classes:

- helicopter
- large_launcher
- large_tower
- medium_launcher
- medium_plane
- mine_roller
- tank

## Change

All 54 cycle-consistent pseudo-labels were manually reviewed.

A label was accepted only when:

- the semantic class was correct
- the propagated bounding box correctly localized the object

Uncertain or incorrect examples were rejected.

## Results

- Reviewed: **54 / 54**
- Accepted: **53**
- Rejected: **1**
- Acceptance rate: **98.1%**

Accepted per class:

| Class | Accepted pseudo-labels |
|---|---:|
| helicopter | 7 |
| large_launcher | 2 |
| large_tower | 12 |
| medium_launcher | 2 |
| medium_plane | 2 |
| mine_roller | 3 |
| tank | 25 |

Combining manual seeds and accepted propagated labels gives up to:

- 23 manual real-domain boxes
- 53 accepted propagated boxes
- **76 real-validation annotations before overlap/deduplication**

## Interpretation

Short-range validation-to-validation propagation is highly reliable when constrained by:

- small temporal gap
- same L1 camera center
- NCC filtering
- forward/backward cycle consistency
- manual verification

The 98.1% acceptance rate is dramatically better than all Helsinki-to-validation transfer approaches.

This establishes real-validation annotation + local propagation as the primary adaptation strategy.

The remaining major weakness is **class coverage**.

Current real-domain labels cover only seven of sixteen classes:

- helicopter
- large_launcher
- large_tower
- medium_launcher
- medium_plane
- mine_roller
- tank

Nine classes remain without validation-domain supervision:

- condor
- hangar
- jammer
- jet_plane
- small_launcher
- small_plane
- small_tower
- spacecraft
- ta-ta

Because competition mAP is macro-averaged across classes, leaving nine classes uncovered is unacceptable.

## Decision

**KEEP — PRIMARY REAL-DOMAIN LABEL GENERATION METHOD**

Next:

1. merge manual and accepted propagated labels into one canonical real-domain annotation set
2. perform a second targeted manual pass focused only on the nine missing classes
3. propagate any new high-confidence seeds locally
4. then train the first validation-domain detector

---

# EXP-D028A — Targeted missing-class validation seed pass

**Status:** COMPLETE

**Date:** 2026-09-17

## Hypothesis

A second targeted manual inspection pass can improve macro-class coverage by focusing exclusively on classes that still lack validation-domain supervision.

## Change

Expanded the validation seed inspection set from 30 to 80 sampled L1 frames.

The annotator was restricted to previously missing classes to reduce distraction from already-covered targets.

## Results

The pass added validation-domain annotations for four previously uncovered classes:

- hangar: 2
- jet_plane: 4
- small_plane: 9
- small_tower: 6

Total canonical annotation file now contains:

- **44 manual annotations**
- **27 labeled frames**

Real-validation class coverage increased from:

- **7 / 16 classes**

to:

- **11 / 16 classes**

Classes still without manual validation seeds:

- condor
- jammer
- small_launcher
- spacecraft
- ta-ta

## Interpretation

Targeted manual inspection is materially improving macro-class coverage.

The newly added classes have enough seeds for direct within-validation propagation testing:

- hangar: 2
- jet_plane: 4
- small_plane: 9
- small_tower: 6

These classes should be benchmarked before pseudo-label generation rather than assumed propagatable.

## Decision

**KEEP**

Proceed to EXP-D028B:

Re-run the leave-one-seed-out local propagation benchmark using the expanded real-validation annotation set and evaluate the four newly seeded classes individually.

---

# EXP-D028B — Expanded-class short-range propagation benchmark

**Status:** COMPLETE

**Date:** 2026-09-17

## Hypothesis

The four newly annotated validation classes may support the same short-range within-domain propagation strategy that previously worked for tank and large_tower.

## Data

Manual validation seeds: **44**

Newly evaluated classes:

- hangar: 2 seeds
- jet_plane: 4 seeds
- small_plane: 9 seeds
- small_tower: 6 seeds

## Aggregate result

Across all classes:

- gap <= 15:
  - pairs: 116
  - R@IoU0.50: 0.388
  - median IoU: 0.015
  - median NCC: 0.986

The aggregate metric is heavily confounded by comparisons between different physical instances of the same semantic class, especially small_plane.

## Per-class short-range results

### hangar

At gap <= 10:

- pairs: 2
- R@IoU0.50: **0.000**
- median IoU: 0.389
- median NCC: 0.851

Even the strongest high-NCC match remains below IoU 0.50.

**Decision:** do not automatically propagate.

### jet_plane

At gap <= 10:

- pairs: 12
- R@IoU0.50: **0.500**
- median NCC: 0.977

At NCC >= 0.90:

- pairs: 11
- P@IoU0.50: **0.545**
- median IoU: **0.715**

Very high NCC does not guarantee correctness; NCC >= 0.98 actually reduces precision to 0.400.

**Decision:** propagate only as AUDIT-ONLY candidates with cycle consistency and human verification.

### small_plane

At gap <= 10:

- pairs: 72
- R@IoU0.50: **0.194**
- median IoU: **0.000**
- median NCC: **0.989**

NCC >= 0.90–0.98 does not materially improve precision.

This is strong evidence that similar-looking same-class instances cause template confusion.

**Decision:** do not trust automatic propagation.

### small_tower

At gap <= 5:

- pairs: 4
- R@IoU0.50: **1.000**
- median IoU: 0.682

At gap <= 10:

- pairs: 12
- R@IoU0.50: **1.000**
- median IoU: 0.686

At gap <= 10 and NCC >= 0.90:

- pairs: 10
- P@IoU0.50: **1.000**
- median IoU: **0.706**

At gap <= 15 and NCC >= 0.94:

- pairs: 8
- P@IoU0.50: **1.000**
- median IoU: 0.706

**Decision:** promote small_tower to TRUSTED propagation class.

## Interpretation

Propagation reliability is strongly class-dependent.

A single global NCC threshold is inappropriate.

Current trusted propagation classes:

- tank
- large_tower
- small_tower

Audit-only propagation:

- jet_plane
- helicopter
- large_launcher
- medium_launcher
- medium_plane
- mine_roller

Manual-only / no automatic propagation:

- hangar
- small_plane

## Decision

**KEEP class-specific propagation policy.**

Proceed to EXP-D028C:
- add small_tower to trusted propagation
- generate cycle-consistent candidates from the expanded 44-seed set
- retain all other propagated classes as human-audit-only
- do not automatically trust jet_plane despite high NCC

---

# EXP-D028C — Expanded cycle-consistent pseudo-label generation

**Status:** COMPLETE

**Date:** 2026-09-17

## Hypothesis

The expanded 44-seed validation annotation set can generate substantially more real-domain pseudo-labels while preserving the high precision observed in EXP-D027D/E.

## Configuration

Validation sequence:

`3224a582bfbf4273a028497662b7aa7c`

Manual seeds:

- 44 annotations
- 27 labeled frames
- 11 / 16 classes covered

Propagation:

- L1 views only
- maximum temporal gap: 10
- same camera center required
- forward NCC >= 0.90
- multi-scale template search
- backward match required
- cycle IoU >= 0.50
- deduplication at IoU >= 0.50

Trusted classes:

- tank
- large_tower
- small_tower

All remaining propagated classes were retained as audit-only.

## Results

- L1 frames available: **194**
- Manual seeds: **44**
- Raw forward matches: **97**
- Cycle-consistent matches: **93**
- Deduplicated pseudo-labels: **82**
- Trusted pseudo-labels: **46**
- Audit-only pseudo-labels: **36**

Per class:

| Class | Candidates | Status |
|---|---:|---|
| hangar | 3 | AUDIT |
| helicopter | 7 | AUDIT |
| jet_plane | 6 | AUDIT |
| large_launcher | 2 | AUDIT |
| large_tower | 12 | TRUST |
| medium_launcher | 2 | AUDIT |
| medium_plane | 2 | AUDIT |
| mine_roller | 4 | AUDIT |
| small_plane | 10 | AUDIT |
| small_tower | 9 | TRUST |
| tank | 25 | TRUST |

## Interpretation

Expanding the real-validation seed set materially increased candidate coverage:

- EXP-D027D: 54 candidates
- EXP-D028C: 82 candidates

Cycle consistency continues to reject very few high-NCC local candidates, while the trusted-class subset remains visually coherent.

Small_tower is now behaving comparably to tank and large_tower and remains appropriate for trusted propagation.

Jet_plane and small_plane generate useful-looking candidates but cannot be automatically trusted because previous pairwise calibration showed substantial instance confusion.

## Decision

**KEEP**

Proceed to EXP-D028D:

- manually review all 82 pseudo-labels
- accept only labels with correct class and localization
- merge accepted pseudo-labels with the 44 manual seeds
- deduplicate the canonical real-domain annotation set
- then train the first real-validation-domain detector

---

# EXP-D028D — Human audit of expanded cycle-consistent pseudo-labels

**Status:** COMPLETE

**Date:** 2026-09-17

## Hypothesis

Cycle-consistent propagation from the expanded 44-seed validation set can generate a substantially larger real-domain training set while retaining high precision after human verification.

## Input

EXP-D028C produced:

- raw forward matches: 97
- cycle-consistent matches: 93
- deduplicated pseudo-labels: 82
- trusted candidates: 46
- audit-only candidates: 36

## Change

All 82 pseudo-label candidates were manually reviewed.

A candidate was accepted only if:

- the semantic class was correct
- the propagated bounding box correctly localized the object

## Results

- Reviewed: **82 / 82**
- Accepted: **79**
- Rejected: **3**
- Acceptance rate: **96.3%**

Accepted per class:

| Class | Accepted |
|---|---:|
| hangar | 3 |
| helicopter | 7 |
| jet_plane | 4 |
| large_launcher | 2 |
| large_tower | 12 |
| medium_launcher | 2 |
| medium_plane | 2 |
| mine_roller | 3 |
| small_plane | 10 |
| small_tower | 9 |
| tank | 25 |

Combined with the 44 manual annotations:

- manual labels: 44
- accepted propagated labels: 79
- **123 real-domain labels before final deduplication**

Real-validation class coverage remains:

- **11 / 16 classes**

Classes still uncovered:

- condor
- jammer
- small_launcher
- spacecraft
- ta-ta

## Interpretation

Cycle-consistent local propagation remains highly reliable after expanding to more classes and more manual seeds.

The acceptance rate decreased only slightly from EXP-D027E:

- EXP-D027E: 98.1%
- EXP-D028D: 96.3%

This confirms that the validation-domain propagation strategy scales beyond the original tank/large-tower subset.

The current dataset is now large enough to train a first real-domain detector and measure performance on held-out manual labels.

## Decision

**KEEP — BUILD FIRST REAL-DOMAIN DETECTOR**

Proceed to EXP-D029:

1. merge manual + accepted pseudo-labels
2. deduplicate overlapping boxes
3. preserve manual labels over pseudo-labels
4. split validation frames temporally into train/holdout
5. train a multi-class detector on real validation imagery
6. evaluate only on held-out real-domain manual labels

---

# EXP-D029 — Canonical real-validation-domain dataset

**Status:** COMPLETE

**Date:** 2026-09-17

## Hypothesis

The manually labeled validation seeds and human-approved propagated pseudo-labels can be merged into a sufficiently clean canonical real-domain dataset to support the first detector trained directly on the validation distribution.

## Input

Manual validation annotations:

- 44 boxes
- 27 labeled frames
- 11 / 16 classes represented

Accepted propagated pseudo-labels from EXP-D028D:

- 79 boxes
- 96.3% human acceptance rate before merge

Raw combined supervision before deduplication:

- 44 manual
- 79 pseudo
- **123 boxes**

Manual annotations were given priority whenever a pseudo-label overlapped the same class/object.

## Change

Built a canonical annotation set by:

1. loading all 44 manual labels
2. loading all 79 accepted propagated labels
3. grouping annotations by frame
4. preserving manual annotations over conflicting pseudo-labels
5. removing same-class duplicates at high IoU
6. retaining the original L1 validation imagery

Initial plan:

- late temporal block for validation
- remaining labeled frames for training

## Results

After deduplication:

- Canonical boxes: **115**
- Manual boxes: **44**
- Pseudo boxes retained: **71**

Initial temporal split:

- Train frames: **63**
- Validation frames: **14**

### Initial training class counts

- hangar: 5
- helicopter: 9
- jet_plane: 6
- large_launcher: 3
- large_tower: 12
- medium_launcher: 3
- medium_plane: 4
- mine_roller: 5
- small_plane: 17
- small_tower: 12
- tank: 22

### Initial validation class counts

- large_launcher: 1
- large_tower: 5
- tank: 11

## Interpretation

The canonical merge succeeded and produced 115 high-confidence real-domain boxes.

However, the initial late temporal holdout was unsuitable for model evaluation because only **3 of the 11 covered classes** appeared in validation:

- large_launcher
- large_tower
- tank

A detector score from this split would therefore provide little information about the remaining covered classes.

A naive random split was also rejected because neighboring pseudo-labels frequently represent the same physical instance and would create severe temporal leakage.

## Decision

**KEEP canonical dataset, REJECT initial split.**

Proceed to EXP-D029A:

- construct a manual-object holdout
- exclude neighboring propagated labels around held-out objects
- maximize class coverage while minimizing same-instance leakage
- evaluate detector recall against manually labeled objects rather than incomplete frame-level annotations

---

# EXP-D029A — Leakage-aware manual-object holdout

**Status:** COMPLETE

**Date:** 2026-09-17

## Hypothesis

A manually selected, leakage-aware holdout can provide a substantially more trustworthy estimate of real-domain detector generalization than either:

- the sparse late temporal block, or
- a random train/validation split.

The holdout should preserve manual target boxes while excluding nearby propagated versions of the same class/object from training.

## Split design

Canonical dataset:

- 115 boxes

Holdout policy:

- use only manually annotated objects as evaluation targets
- exclude all annotations from held-out frames
- exclude pseudo-labels of the same class within ±10 frame indices of each held-out manual object
- preserve as much remaining training supervision as possible

Initial automated holdout selection produced:

- Training boxes: 50
- Training frames: 41
- Holdout manual boxes: 14
- Holdout frames: 12
- Nearby pseudo-labels excluded: 39

However, this first selection left:

- hangar: 0 training examples
- large_launcher: 0 training examples

and was therefore rejected before training.

## Holdout candidate-cost analysis

Every manually labeled candidate was evaluated for how many same-class training examples would be lost under the ±10-frame leakage exclusion.

This showed that several holdout choices were substantially cheaper than others.

Examples:

- large_launcher frame 162:
  - remaining training examples: 1

- large_launcher frame 205:
  - remaining training examples: 3

- helicopter frame 66:
  - remaining: 4

- helicopter frame 107:
  - remaining: 5

- mine_roller frame 14:
  - remaining: 4

- mine_roller frame 92:
  - remaining: 1

The holdout was therefore changed from automatic evenly spaced selection to explicit manually chosen objects.

## Final explicit holdout

Selected manual objects:

### Frame 125

- hangar
- jet_plane
- small_plane
- small_plane

### Frame 107

- helicopter
- large_tower

### Frame 205

- large_launcher
- tank

### Frame 82

- medium_plane
- tank

### Frame 14

- mine_roller

### Frame 182

- small_tower

### Frame 186

- small_tower

### Frame 222

- large_tower

This yields:

- **14 held-out manual objects**
- **8 held-out frames**
- **10 evaluated classes**

## Final split results

- Canonical boxes: **115**
- Training boxes: **67**
- Training frames: **54**
- Holdout manual boxes: **14**
- Holdout frames: **8**
- Pseudo-labels excluded near holdout: **31**

### Training counts

- hangar: 1
- helicopter: 5
- jet_plane: 3
- large_launcher: 3
- large_tower: 11
- medium_launcher: 2
- medium_plane: 1
- mine_roller: 4
- small_plane: 6
- small_tower: 8
- tank: 23

### Manual holdout counts

- hangar: 1
- helicopter: 1
- jet_plane: 1
- large_launcher: 1
- large_tower: 2
- medium_plane: 1
- mine_roller: 1
- small_plane: 2
- small_tower: 2
- tank: 2

## Interpretation

The revised holdout is substantially more useful than the original temporal block.

Advantages:

- 10 covered classes are evaluated
- only 8 frames are removed
- nearby pseudo-label leakage is explicitly reduced
- strong classes retain substantial training supervision
- large_launcher and helicopter now retain usable training data

Two classes remain intrinsically data-limited:

- hangar: 1 training example
- medium_plane: 1 training example

Their holdout results must therefore be interpreted cautiously.

The holdout is intentionally object-centric rather than conventional full-frame mAP evaluation because the retained validation imagery is incompletely annotated.

A real but unlabeled detection should not be automatically counted as a false positive.

## Decision

**KEEP — USE AS AUTHORITATIVE INTERNAL HOLDOUT**

EXP-D029B will train on the 67 leakage-reduced training boxes.

EXP-D029C will evaluate the trained detector only against the 14 manually held-out objects using correct-class IoU >= 0.50.

---

# EXP-D029B — First real-validation-domain detector

**Status:** COMPLETE

**Date:** 2026-09-17

## Hypothesis

A YOLO11n detector fine-tuned directly on clean real-validation-domain supervision should generalize substantially better to unseen validation objects than the previous Helsinki-trained detector.

## Training data

Leakage-aware EXP-D029A training set:

- Training frames: **54**
- Training boxes: **67**
- Classes represented in training: **11**

Training class counts:

- hangar: 1
- helicopter: 5
- jet_plane: 3
- large_launcher: 3
- large_tower: 11
- medium_launcher: 2
- medium_plane: 1
- mine_roller: 4
- small_plane: 6
- small_tower: 8
- tank: 23

Classes still completely absent from real-domain supervision:

- condor
- jammer
- small_launcher
- spacecraft
- ta-ta

## Model

Base model:

- `yolo11n.pt`

Device:

- Apple M1 MPS

Training configuration:

- image size: 960
- batch size: 4
- epochs: 80
- pretrained weights: yes
- horizontal flip: 0.5
- vertical flip: 0.0
- rotation: 3 degrees
- translation: 0.05
- scale augmentation: 0.15
- mosaic: 0.25
- mixup: 0.0
- copy-paste: 0.0
- seed: 42

## Training setup correction

Ultralytics requires detection dataset YAML files to contain both `train:` and `val:` entries.

The first training launch failed before optimization because the leakage-aware dataset intentionally contained no conventional validation split.

The YAML was therefore given:

`val: images/train`

only to satisfy Ultralytics dataset-schema requirements.

The authoritative evaluation remains the independent manual holdout from EXP-D029A.

## Training result

Training completed successfully.

Final model:

`drone/artifacts/exp_d029b/runs/yolo11n_validation_domain/weights/last.pt`

Model:

- YOLO11n
- parameters: **2,585,272**
- GFLOPs: **6.4**

Ultralytics reported the following metrics on the 54 training images:

- Precision: **0.859**
- Recall: **0.878**
- mAP@0.50: **0.859**
- mAP@0.50:0.95: **0.661**

Per-class training-set mAP@0.50:

- hangar: 0.497
- helicopter: 0.995
- jet_plane: 0.995
- large_launcher: 0.995
- large_tower: 0.995
- medium_launcher: 0.190
- medium_plane: 0.995
- mine_roller: 0.995
- small_plane: 0.924
- small_tower: 0.872
- tank: 0.995

Reported inference speed during this internal pass:

- preprocess: 0.3 ms/image
- inference: 7.3 ms/image
- postprocess: 13.0 ms/image

## Interpretation

The model clearly fit most of the real-domain training classes, which confirms that the 67-box dataset is learnable.

However, these Ultralytics metrics are **resubstitution metrics on the training images** and must not be interpreted as evidence of real-domain generalization.

Several effects are visible even on the training set:

- medium_launcher remains weak, consistent with only two training examples
- hangar remains weak, consistent with only one training example
- most better-represented classes are fit nearly perfectly

The decisive question is now whether the detector generalizes to the 14 manually held-out objects that were explicitly protected from local pseudo-label leakage.

## Decision

**KEEP — PROCEED TO INDEPENDENT HOLDOUT EVALUATION**

EXP-D029C is the authoritative model-selection experiment.

---

# EXP-D029C — Leakage-aware real-domain detector holdout evaluation

**Status:** COMPLETE

**Date:** 2026-09-17

## Hypothesis

If validation-domain manual annotation and cycle-consistent pseudo-label propagation successfully solve the domain-shift problem, the EXP-D029B detector should recover a substantial fraction of genuinely held-out real-validation objects.

## Model

Checkpoint:

`drone/artifacts/exp_d029b/runs/yolo11n_validation_domain/weights/last.pt`

Model:

- YOLO11n
- trained directly on real validation-domain imagery
- 67 training boxes
- 54 training frames
- 11 represented classes

## Holdout

Leakage-aware manual holdout from EXP-D029A:

- **14 manually annotated objects**
- **8 held-out frames**
- **10 classes**
- neighboring same-class pseudo-labels within ±10 frames excluded from training

Success criterion:

- correct semantic class
- IoU >= 0.50

Confidence thresholds tested:

- 0.001
- 0.005
- 0.01
- 0.025
- 0.05
- 0.10
- 0.20

## Results

Best operating point:

- Confidence: **0.001**
- Recall @ IoU0.50: **0.786**
- Correctly recovered held-out objects: **11 / 14**
- Mean proposals/frame: **42.4**
- Median inference latency: **17.5 ms**

Results:

`drone/artifacts/exp_d029c/holdout_results.json`

## Interpretation

The detector generalizes strongly beyond the real-domain objects used for training.

Recovering 11 of 14 manually held-out objects after explicit temporal leakage reduction provides much stronger evidence of useful generalization than the earlier Helsinki-based experiments.

This validates the central pivot of the project:

`Helsinki supervision -> validation domain`

was weak, while:

`manual validation seeds -> local propagation -> verified pseudo-labels -> validation-domain detector`

is effective.

The detector is also computationally practical.

At the best recall operating point:

- median model latency is only 17.5 ms
- 42.4 predictions/frame remain manageable for post-processing
- there is substantial headroom relative to the 3333 ms request timeout

The remaining major limitation is class coverage.

Five of sixteen competition classes still have no real-domain supervision:

- condor
- jammer
- small_launcher
- spacecraft
- ta-ta

Therefore the detector's overall competition ceiling is still limited even though performance on represented classes is promising.

## Decision

**KEEP — FIRST SUCCESSFUL REAL-DOMAIN DETECTOR**

EXP-D029 establishes validation-domain annotation and training as the primary strategy.

Next priorities:

1. inspect the three missed holdout objects and per-class recall
2. determine whether a slightly higher confidence threshold preserves recall while reducing predictions/frame
3. integrate the real-domain detector into the competition endpoint
4. run another real validation attempt before evaluation
5. continue targeted annotation for the five still-uncovered classes if time permits

---

---

# EXP-D032 — Full real-domain detector training

**Status:** COMPLETE

**Date:** 2026-09-18

## Hypothesis

Training on the full canonical real-validation-domain dataset should improve over the leakage-reduced EXP-D029B model once the holdout experiment has already established that the domain-adaptation strategy generalizes.

## Data

Canonical real-domain dataset:

- Frames: 77
- Boxes: 115
- Classes represented: 11 / 16

Class counts:

- hangar: 5
- helicopter: 9
- jet_plane: 6
- large_launcher: 4
- large_tower: 17
- medium_launcher: 3
- medium_plane: 4
- mine_roller: 5
- small_plane: 17
- small_tower: 12
- tank: 33

Still uncovered:

- condor
- jammer
- small_launcher
- spacecraft
- ta-ta

## Models

### YOLO11n

Validation metrics on full real-domain dataset:

- Precision: 0.949
- Recall: 0.961
- mAP@0.50: 0.967
- mAP@0.50:0.95: 0.798

### YOLO11s

Validation metrics:

- Precision: 0.961
- Recall: 0.955
- mAP@0.50: 0.968
- mAP@0.50:0.95: 0.842

YOLO11s achieved better resubstitution localization quality, especially at higher IoU thresholds.

## Interpretation

Both models fit the validation-domain supervision extremely well.

However, earlier experiments established that internal training/validation metrics are not sufficient for model selection. Remote competition validation remains authoritative.

YOLO11n was selected first for deployment because it is substantially lighter and leaves more realtime headroom.

## Decision

**KEEP both checkpoints.**

Deploy YOLO11n first and compare remotely before adopting the larger model.

---

# EXP-D033 — Unique-view / missing-class search

**Status:** COMPLETE / NEGATIVE RESULT

**Date:** 2026-09-18

## Hypothesis

Additional retained validation views may expose the five classes still missing from real-domain supervision:

- condor
- jammer
- small_launcher
- spacecraft
- ta-ta

## Change

Constructed unique-view pools and contact sheets from retained validation captures to search for visually distinct regions and previously unseen targets.

## Result

Manual inspection did not reveal reliable examples of the five missing classes.

The generated contact sheets largely contained the same visual regions and repeated validation imagery already inspected.

## Interpretation

The remaining missing classes cannot be recovered cheaply through broader contact-sheet sampling of the currently captured validation data.

Further manual searching has low expected value compared with improving the detector on the 11 represented classes.

## Decision

**STOP broad missing-class search for now.**

---

# EXP-D034 — Full real-domain model remote deployment

**Status:** COMPLETE

**Date:** 2026-09-18

## Hypothesis

The full 115-box real-domain YOLO11n model should outperform the earlier leakage-reduced D029 model on the official validation sequence.

## Runtime

- model: EXP-D032 YOLO11n
- checkpoint: `last.pt`
- confidence: 0.001
- NMS IoU: 0.70
- imgsz: 960
- fixed L1 centered view
- Mac + ngrok deployment

## Remote validation results

Representative clean runs:

- 0.1271957815
- 0.1211453220
- 0.1402458666
- 0.1230270320
- 0.1237819620
- 0.1426608582
- 0.1227743876
- 0.1259454492
- 0.1231092412
- 0.1259796970
- 0.1163018758
- 0.1267514088

Best observed ngrok-era result:

**0.1426608582**

## Interpretation

The full real-domain detector is clearly better than the earlier Helsinki-based systems.

However, score variance remained unexpectedly large even when model/runtime settings were unchanged.

Remote telemetry later showed that the Mac + ngrok endpoint was receiving only roughly 175 / 249 requests on some runs.

Therefore model selection during this phase was heavily confounded by transport/frame-loss instability.

## Decision

**KEEP D032 YOLO11n.**

Do not trust fine-grained remote parameter comparisons until deployment transport is fixed.

---

# EXP-D035 — Direct AWS Stockholm deployment

**Status:** COMPLETE — MAJOR INFRASTRUCTURE IMPROVEMENT

**Date:** 2026-09-18

## Hypothesis

The Mac + ngrok deployment is dropping a substantial fraction of realtime frames.

Moving the same detector and camera policy to a directly reachable cloud VM in Stockholm should improve request coverage and score even if CPU inference is slower.

## Infrastructure

AWS EC2:

- Region: Europe (Stockholm), `eu-north-1`
- Instance: `m7i-flex.large`
- 2 vCPU
- 8 GiB RAM
- Intel Xeon Platinum 8488C
- Ubuntu 24.04 x86_64
- direct public IPv4
- no ngrok

## Model

Frozen configuration:

- D032 YOLO11n
- confidence: 0.001
- NMS IoU: 0.70
- imgsz: 960
- fixed L1 center `(1920,1080)`

## CPU benchmark

YOLO11n @ 960:

- mean: 99.3 ms
- median: 99.2 ms
- p95: 103.2 ms
- max: 106.6 ms

## Remote validation

Run 1:

- score: **0.1640706308**
- coverage: **248 / 249**
- median endpoint latency: **109.8 ms**
- max endpoint latency: **140.1 ms**

Run 2:

- score: **0.1660387593**

Comparable Mac + ngrok run:

- score: 0.1267514088
- coverage: 175 / 249

## Interpretation

Transport was a major bottleneck.

The AWS CPU endpoint was slower than Mac inference but delivered almost every frame.

Coverage improved from approximately:

- 70.3%
to
- 99.6%

This produced a large score increase.

## Decision

**KEEP AWS direct deployment.**

Treat AWS as the clean reference environment for remote model selection.

---

# EXP-D036 — AWS confidence sweep

**Status:** COMPLETE

**Date:** 2026-09-18

## Baseline

Confidence 0.001:

- 0.1640706308
- 0.1660387593

Mean:

**0.1650546951**

## Confidence 0.005

Runs:

- 0.1630635797
- 0.1638353306

Mean:

**0.1634494552**

## Interpretation

Once transport noise was removed, confidence 0.001 consistently outperformed 0.005.

## Decision

**KEEP confidence = 0.001**

---

# EXP-D037 — NMS sweep

**Status:** COMPLETE

**Date:** 2026-09-18

## Results

### NMS IoU 0.50

- score: 0.1597725318
- coverage: 246 / 249
- median latency: 97.7 ms

### NMS IoU 0.60

- score: 0.1644263785
- coverage: 249 / 249
- median latency: 102.1 ms

### NMS IoU 0.70

Baseline mean:

**~0.16505**

## Interpretation

0.50 is clearly too aggressive.

0.60 is competitive but does not beat the 0.70 baseline.

## Decision

**KEEP NMS IoU = 0.70**

---

# EXP-D038 — Fixed-camera center sweep

**Status:** COMPLETE

**Date:** 2026-09-18

## Hypothesis

A different fixed L1 crop may expose more useful target content than the centered view.

## Results

| L1 center | Remote score |
|---|---:|
| `(1920,1080)` | **0.16407 / 0.16604** |
| `(2880,540)` | 0.146887 |
| `(2240,1080)` | 0.136713 |
| `(2880,1620)` | 0.111058 |
| `(960,540)` | 0.088297 |
| `(960,1620)` | 0.085306 |

All major off-center runs had near-perfect frame delivery.

## Interpretation

The centered L1 view is decisively superior.

The score losses are spatial-content effects, not transport artifacts.

## Decision

**KEEP fixed center `(1920,1080)`**

Stop broad camera-center search.

---

# EXP-D039 — Increased inference resolution

**Status:** COMPLETE

**Date:** 2026-09-18

## Hypothesis

Increasing inference resolution from 960 to 1280 may improve small-object localization while remaining inside the realtime budget.

## AWS benchmark at imgsz=1280

- mean: 175.8 ms
- median: 176.3 ms
- p95: 182.8 ms
- max: 183.7 ms

## Remote validation

- score: **0.1491883519**
- coverage: 248 / 249
- median endpoint latency: 200.5 ms
- max endpoint latency: 220.7 ms

## Interpretation

The larger inference scale remained realtime-safe but substantially reduced validation score.

This is therefore a model-scale/generalization regression rather than a latency problem.

## Decision

**REJECT imgsz=1280**

Restore:

`IMAGE_SIZE = 960`

---

# EXP-D040 — Multi-seed YOLO11n training

**Status:** COMPLETE

**Date:** 2026-09-18

## Hypothesis

With only 115 real-domain boxes, random initialization / augmentation order may materially affect hidden-domain generalization.

## Seeds trained

- 7
- 21
- 1337

Original baseline:

- 42

## Internal training metrics

| Seed | mAP@0.50 | mAP@0.50:0.95 |
|---:|---:|---:|
| 7 | 0.9663 | **0.8012** |
| 21 | 0.9653 | 0.7998 |
| 1337 | **0.9689** | 0.7933 |
| 42 | ~0.9670 | ~0.7980 |

## Remote validation

| Seed | Score |
|---:|---:|
| 42 | ~0.16505 mean |
| 1337 | 0.1715091434 |
| 7 | 0.1720154731 |
| **21** | **0.1736260883** |

All three new-seed runs achieved essentially complete frame delivery.

## Interpretation

Training-seed variance materially affects hidden-validation performance.

Seed 21 generalizes best despite not having the strongest internal mAP metric.

This again confirms that internal resubstitution metrics are insufficient for final model selection.

## Decision

**KEEP seed 21 as strongest single-model detector.**

---

# EXP-D041 — Seed 21 + seed 7 ensemble

**Status:** COMPLETE

**Date:** 2026-09-18

## Hypothesis

The two strongest independently trained YOLO11n models may make complementary errors.

Combining their detections and suppressing same-class duplicates may improve hidden-validation performance.

## Models

- seed 21
- seed 7

## Initial ensemble

- confidence: 0.001
- per-model NMS IoU: 0.70
- ensemble class-aware NMS IoU: 0.55
- imgsz: 960

AWS CPU benchmark:

- mean: 231.5 ms
- median: 233.0 ms
- p95: 239.5 ms
- max: 239.7 ms

Remote validation:

- run 1: **0.1739473919**
- run 2: **0.1739473919**

The repeated run produced the exact same score to displayed precision.

## Ensemble NMS sweep

### 0.55

- score: 0.1739473919

### 0.65

- score: **0.1752288579**

### 0.75

- score: 0.1738262207
- coverage: 248 / 249
- median endpoint latency: 281.4 ms
- max: 493.7 ms

## Interpretation

The two-model ensemble improves over the strongest individual seed.

An ensemble NMS threshold of 0.65 preserves useful complementary detections better than 0.55 while avoiding the excess duplicate load seen at 0.75.

## Decision

**KEEP D041B**

Production ensemble:

- seed 21 + seed 7
- confidence: 0.001
- per-model NMS: 0.70
- ensemble NMS: **0.65**
- imgsz: 960
- center: `(1920,1080)`

Best remote score:

**0.1752288579**

---

# EXP-D042 — Direct Windows GTX 1060 deployment

**Status:** COMPLETE

**Date:** 2026-09-18

## Hypothesis

A directly exposed home Windows endpoint may preserve near-complete frame delivery while providing substantially faster CUDA inference than the AWS CPU instance.

## Hardware

- NVIDIA GeForce GTX 1060 6 GB
- PyTorch 2.14.0 + CUDA 12.6
- direct public IPv4
- no ngrok

## D041B local GPU benchmark

Seed 21 + seed 7 ensemble:

- mean: 22.2 ms
- median: 22.0 ms
- p95: 23.9 ms
- max: 24.5 ms

## Remote validation

- score: **0.1750922118**
- coverage: **248 / 249**
- missing frame: `[150]`
- median endpoint latency: **47.4 ms**
- max endpoint latency: **182.3 ms**

AWS reference:

- score: 0.1752288579

## Interpretation

Windows direct hosting is effectively score-equivalent to AWS while providing dramatically lower runtime latency and much more compute headroom.

This makes Windows the preferred platform for compute-heavy experimentation.

AWS remains the strongest validated reference deployment.

## Decision

**KEEP Windows direct as primary experimentation platform.**

---

# EXP-D043A — Three-model ensemble

**Status:** COMPLETE

**Date:** 2026-09-18

## Configuration

Models:

- seed 21
- seed 7
- seed 1337

Runtime:

- confidence: 0.001
- per-model NMS IoU: 0.70
- ensemble NMS IoU: 0.65
- imgsz: 960
- camera: centered L1 `(1920,1080)`
- Windows GTX 1060 direct endpoint

## Local benchmark

- mean: 32.5 ms
- median: 31.6 ms
- p95: 35.3 ms
- max: 39.1 ms

## Remote validation

- score: **0.1739219290**
- coverage: **247 / 249**
- missing frames: `[105, 106]`
- median endpoint latency: **61.8 ms**
- max endpoint latency: **140.9 ms**
- errors: none

## Comparison

- seed 21 single: 0.1736260883
- seed 21 + 7 ensemble: **0.1752288579**
- seed 21 + 7 + 1337: 0.1739219290

## Interpretation

Adding seed 1337 does not improve hidden-validation performance.

The third model likely contributes enough weaker or redundant predictions to offset any benefit from additional diversity.

## Decision

**REJECT the 3-model ensemble.**

Retain seed 21 + seed 7 with ensemble NMS 0.65 as the current best.
# EXP-D044A — Score-weighted box fusion

**Status:** COMPLETE

**Date:** 2026-09-18

## Hypothesis

The seed-21 + seed-7 ensemble may benefit from averaging overlapping same-class
boxes rather than keeping only the highest-confidence prediction.

## Change

Replaced hard cross-model class-aware NMS with score-weighted coordinate fusion.

Fixed:

- seed 21 + seed 7
- confidence: 0.001
- per-model NMS IoU: 0.70
- fusion IoU: 0.65
- imgsz: 960
- camera: centered L1 `(1920,1080)`
- Windows GTX 1060 direct endpoint

## Local benchmark

- mean: 23.2 ms
- median: 23.3 ms
- p95: 25.3 ms
- max: 25.8 ms

## Remote validation

- score: **0.1748360702**
- coverage: **248 / 249**
- missing frame: `[1]`
- median endpoint latency: **57.5 ms**
- max endpoint latency: **141.8 ms**
- errors: none

## Comparison

- hard NMS ensemble: **0.1752288579**
- weighted box fusion: 0.1748360702

## Interpretation

Weighted coordinate averaging slightly reduces hidden-validation performance.

The two models are complementary, but when both predict the same object the
higher-confidence box is often better localized than the averaged box.

## Decision

**REJECT WBF.**

Restore D041B hard class-aware NMS at ensemble IoU 0.65.

---


# EXP-D045 — Real + synthetic model / missing-class specialist

## EXP-D045B — Seed-21 real + synthetic model

Checkpoint:

`drone/artifacts/exp_d045b/runs/seed21_real_plus_synth_v1-2/weights/best.pt`

Local validation:

- images: `2970`
- instances: `3350`
- overall precision: `0.843`
- recall: `0.930`
- mAP50: `0.876`
- mAP50-95: `0.762`

Selected class results included:

- condor mAP50: `0.941`
- helicopter mAP50: `0.880`
- jammer mAP50: `0.935`
- mine_roller mAP50: `0.970`
- spacecraft mAP50: `0.955`
- tank mAP50: `0.791`

### Remote single-model test

The runtime was temporarily configured with the same D045B checkpoint in both ensemble slots.

Remote result:

- score: `0.1302117671`
- errors: none
- remote requests: `249/249`
- median latency: `46.6 ms`
- max latency: `107.3 ms`

### Interpretation

Strong local validation did not transfer to the hidden remote sequence.

The original D040 ensemble remained substantially stronger remotely.

### Decision

Do not replace D041B with D045B as the main detector.

---

## EXP-D045C — Missing-class specialist integration

A specialist detector was added for classes believed to be underrepresented:

- `condor`
- `jammer`
- `small_launcher`
- `spacecraft`
- `ta-ta`

The specialist replaced baseline predictions for the target class set.

### Remote tests

Initial specialist integration:

- score: `0.1745149012`
- telemetry: `248/249`
- median latency: `60.7 ms`
- max latency: `171.5 ms`

Specialist threshold variant:

- score: `0.1650766644`
- telemetry: `230/249`
- missing: `19`
- median latency: `62.7 ms`

Specialist confidence `0.05`:

- score: `0.1752240687`
- telemetry: `247/249`
- missing: `2`
- median latency: `56.4 ms`
- max latency: `200.8 ms`

### Interpretation

The specialist did not produce a clear reproducible gain over D041B.

Later inspection of the real validation imagery suggested that the supposedly missing classes might not occur in this sequence at all.

### Decision

Do not rely on the missing-class specialist as the main improvement path.

---

# EXP-D047 — Real-domain pseudo-label mining

## EXP-D047A — Two-model agreement mining

The D040 seed-21 and seed-7 models were run independently over the captured L1 frames.

Input:

- L1 frames: `194`

Mining output:

- total agreements: `970`
- strong agreements: `248`
- disagreements: `4706`
- review candidates: `548`

Strong-agreement class counts:

- hangar: `11`
- helicopter: `16`
- jet_plane: `15`
- large_launcher: `7`
- large_tower: `35`
- medium_launcher: `8`
- medium_plane: `19`
- mine_roller: `11`
- small_plane: `46`
- small_tower: `15`
- tank: `65`

Agreement statistics:

- median minimum confidence: `0.8371`
- minimum minimum-confidence: `0.0201`
- maximum minimum-confidence: `0.9947`
- median IoU: `0.9102`
- IoU range: `0.6020–0.9783`

Threshold study:

- conf >= `0.5`, IoU >= `0.75`: `160`
- conf >= `0.7`, IoU >= `0.80`: `147`
- conf >= `0.8`, IoU >= `0.85`: `128`
- conf >= `0.9`, IoU >= `0.90`: `64`

Balanced selection at conf >= `0.70`, IoU >= `0.80`:

- selected: `105`

Class distribution:

- hangar: `3`
- helicopter: `10`
- jet_plane: `8`
- large_launcher: `6`
- large_tower: `12`
- medium_launcher: `5`
- medium_plane: `8`
- mine_roller: `10`
- small_plane: `12`
- small_tower: `11`
- tank: `20`

---

## EXP-D047B — Pseudo-labeled real-domain dataset

Dataset build result:

- new D047 images: `83`
- new D047 boxes: `105`
- total images: `160`
- total label files: `160`
- total boxes: `220`

Class counts:

- hangar: `8`
- helicopter: `19`
- jet_plane: `14`
- large_launcher: `10`
- large_tower: `29`
- medium_launcher: `8`
- medium_plane: `12`
- mine_roller: `15`
- small_plane: `29`
- small_tower: `23`
- tank: `53`

---

## EXP-D047C — Retraining on mined real-domain dataset

Training performed on Linux GTX 1060 3 GB.

Example seed-7 local validation:

- images: `160`
- instances: `220`
- precision: `0.969`
- recall: `0.927`
- mAP50: `0.969`
- mAP50-95: `0.840`

Selected class mAP50-95:

- hangar: `0.893`
- helicopter: `0.909`
- jet_plane: `0.813`
- large_launcher: `0.902`
- large_tower: `0.866`
- medium_launcher: `0.886`
- medium_plane: `0.858`
- mine_roller: `0.894`
- small_plane: `0.622`
- small_tower: `0.770`
- tank: `0.824`

### Remote result

D047C single-model remote score:

- `0.1595087503`

Telemetry:

- `238/249`
- missing: `11`
- median latency: `50.8 ms`
- max latency: `144.2 ms`

### Interpretation

Excellent local validation again failed to translate into remote score.

Pseudo-label agreement training was not enough to beat D041B.

### Decision

Reject D047C as production replacement.

---

# EXP-D048 — Manual disagreement mining

## EXP-D048A — Hard-disagreement review

Initial selection:

- selected: `19`

Class counts:

- hangar: `2`
- jet_plane: `1`
- large_tower: `2`
- medium_launcher: `1`
- medium_plane: `2`
- mine_roller: `1`
- small_plane: `4`
- small_tower: `2`
- tank: `4`

By source:

- seed21_only: `7`
- seed7_only: `12`

After lowering minimum score to `0.03`:

- selected: `46`

Class counts:

- hangar: `4`
- helicopter: `2`
- jet_plane: `3`
- large_launcher: `1`
- large_tower: `4`
- medium_launcher: `2`
- medium_plane: `4`
- mine_roller: `4`
- small_plane: `9`
- small_tower: `5`
- tank: `8`

By source:

- seed21_only: `23`
- seed7_only: `23`

Manual review accepted `31` candidates.

---

## EXP-D048B — Expanded manually verified dataset

Build result:

- accepted D048 candidates: `31`
- actually added: `27`
- duplicate skips: `4`
- new frames copied: `11`
- total images: `171`
- total boxes: `247`

---

## EXP-D048C — Retraining on disagreement-enriched dataset

Seed-21 local validation:

- images: `171`
- instances: `247`
- precision: `0.975`
- recall: `0.904`
- mAP50: `0.961`
- mAP50-95: `0.817`

Selected class mAP50-95:

- hangar: `0.880`
- helicopter: `0.878`
- jet_plane: `0.761`
- large_launcher: `0.917`
- large_tower: `0.840`
- medium_launcher: `0.882`
- medium_plane: `0.810`
- mine_roller: `0.902`
- small_plane: `0.585`
- small_tower: `0.693`
- tank: `0.844`

### Interpretation

Manual disagreement labeling improved the real-domain dataset quality but did not expose a clear route toward the leaderboard gap.

---

# EXP-D049 — Exhaustive L1 visual review

All `194` captured L1 frames were split into quadrants for manual inspection.

Generated:

- L1 frames: `194`
- detail tiles: `776`
- sheets: `194`

Output:

`drone/artifacts/exp_d049/quadrant_sheets`

### Purpose

Search manually for classes not represented in the mined real-domain dataset, especially:

- condor
- jammer
- small_launcher
- spacecraft
- ta-ta

### Observation

Extensive manual review did not reveal convincing examples of these missing classes.

### Interpretation

The working assumption that these classes were “missing because the detector had not learned them” became increasingly doubtful.

They may simply not occur in the sampled validation sequence.

---

# EXP-D050 — Missing-class specialist scan

The D045 specialist was run across the captured real validation frames specifically to propose:

- condor
- jammer
- small_launcher
- spacecraft
- ta-ta

### Result

Contact-sheet review showed that the proposals were overwhelmingly false positives.

The specialist frequently mapped ordinary terrain, buildings, vegetation, or image artifacts to the supposedly missing classes.

### Interpretation

This strongly suggested that forcing the five missing classes into the real-domain dataset was harmful.

### Decision

Stop pseudo-labeling or synthetically forcing these classes without clear visual evidence.

---

# EXP-D050B — Class suppression test

The following classes were suppressed at runtime:

- condor
- jammer
- small_launcher
- spacecraft
- ta-ta

Baseline detector remained:

- D040 seed 21 + seed 7
- confidence `0.001`
- model NMS `0.70`
- ensemble NMS `0.65`
- imgsz `960`

Remote runs:

### Run 1

- score: `0.1717109464`
- received: `242/249`
- missing: `7`
- median latency: `81.1 ms`

### Run 2

- score: `0.1752366674`
- received: `248/249`
- missing: `1`
- median latency: `79.4 ms`

### Run 3

- score: `0.1738982910`
- received: `247/249`
- missing: `2`
- median latency: `80.6 ms`

### Interpretation

The apparent `0.1752367` improvement was not reproducible.

The filtered runs averaged below the original D041B baseline.

### Decision

Reject class suppression as a meaningful gain.

Restore D041B.

---

# EXP-D051 — Active camera investigation

## EXP-D051A — L1 target-following

Instead of fixed centered L1, the camera attempted to move within L1 toward small/high-confidence detections.

Remote score:

- `0.1784109350`

This became the highest observed score during this sequence of experiments.

### Important caveat

Log inspection showed that the camera largely remained at centered L1, so the score increase could not confidently be attributed to the targeting policy itself.

### Decision

Preserve the configuration as a useful checkpoint, but do not interpret the gain as proof of successful camera tracking.

---

# EXP-D052 — L2 active zoom investigation

## EXP-D052A — L1 detection → L2 target zoom

Camera constraints were re-examined.

Key discovery:

- from L0, allowed levels: `[0, 1]`
- from L1, allowed levels: `[0, 1, 2]`

Therefore L2 was legal when requested from L1.

### First L2 attempt

The camera successfully entered L2 on a high-confidence `mine_roller`.

Example:

- class: `mine_roller`
- confidence: approximately `0.894–0.916`
- L2 center around `(1974,1303)`

However, the runtime attempted an invalid L2 → L0 reset.

The evaluator ignored the reset, leaving the camera stuck at the same L2 crop for nearly the entire sequence.

Remote score:

- `0.0122286713`

### Interpretation

This was a camera-state bug, not evidence that L2 itself was inherently bad.

---

## EXP-D052B — L2 → L1 reset

The reset path was corrected to:

`L2 -> L1`

instead of:

`L2 -> L0`

Remote score:

- `0.03585007345`

Errors showed movement-limit violations:

example:

`551.24 px exceeds the L2 limit of 551.00 px`

### Interpretation

Rounding at the exact movement boundary caused otherwise valid reset commands to be rejected.

---

## EXP-D052C — L2 movement safety margin

A small safety margin was added below the movement limit.

Later remote result:

- score: `0.1209545109`
- received: `233/249`
- missing: `16`
- median latency: `79.4 ms`
- max latency: `176.5 ms`

Camera errors still occurred on several L2 → L1 transitions, including movement distances around:

- `608.04 px`
- `558.92 px`
- `636.95 px`
- `654.28 px`
- `552.15 px`
- `587.59 px`
- `584.58 px`

### Interpretation

The L2 camera controller remained unreliable and materially reduced evaluation coverage and score.

### Decision

Pause active-L2 work.

Return focus to detector/post-processing and captured validation analysis.

---

# EXP-D053 — Hidden-validation capture and post-processing analysis

## EXP-D053A — Capture actual evaluator imagery

The stable centered-L1 predictor was restored.

Validation requests were captured locally using the existing `capture_request()` infrastructure.

Captured sequence:

`drone/captures/ed590596961d4390b6919a7633824f6b`

Captured:

- `230` PNG images
- `230` frame metadata JSON files
- `230` prediction JSON files

Remote score during capture run:

- `0.1578304149`
- received: `230/249`
- missing: `19`
- median latency: `82.5 ms`
- max latency: `202.1 ms`

Medical training was running concurrently on the Windows machine, so this score was considered confounded by compute contention and capture I/O.

### Decision

Use this attempt as a data-acquisition run, not as a model-quality result.

---

## EXP-D053B — Manual validation review

A review generator was created:

`drone/scripts/make_validation_review.py`

It projected the model's global normalized predictions back onto the received validation crops and generated contact sheets.

### Major observation

At the existing production confidence:

`CONFIDENCE = 0.001`

the detector produced massive amounts of low-confidence clutter:

- false positives on trees
- false positives on roads
- false positives on water
- giant hangar-like boxes
- repeated overlapping detections
- many predictions displayed near confidence `0.00`

The captured frames nevertheless also showed strong, temporally persistent high-confidence detections.

---

## EXP-D053C — Confidence distribution analysis

Across `230` captured prediction files:

| Confidence threshold | Total predictions | Average/frame |
|---|---:|---:|
| 0.001 | 7715 | 33.5 |
| 0.005 | 1538 | 6.7 |
| 0.010 | 958 | 4.2 |
| 0.020 | 646 | 2.8 |
| 0.030 | 532 | 2.3 |
| 0.050 | 439 | 1.9 |
| 0.080 | 386 | 1.7 |
| 0.100 | 357 | 1.6 |
| 0.150 | 322 | 1.4 |
| 0.200 | 301 | 1.3 |

### Interpretation

The production threshold `0.001` appears far too permissive for the hidden validation domain.

Raising the threshold from `0.001` to `0.03` removes more than 90% of returned boxes while preserving many strong temporally stable tracks.

---

## EXP-D053D — High-confidence track inspection

A survivor dump was generated for detections with confidence >= `0.020`.

Several strong persistent tracks were observed:

- `mine_roller` around frames 3–9:
  - approximately `0.43–0.99`
- `tank` around frames 25–40:
  - approximately `0.85–0.98`
- `helicopter` around frames 52–64:
  - approximately `0.90–0.99`
- `hangar` around frames 65–76:
  - approximately `0.85–0.99`
- `medium_plane` around frames 70–82:
  - approximately `0.92–0.99`
- `tank` around frames 84–91:
  - approximately `0.89–0.99`
- `helicopter` around frames 93–108:
  - approximately `0.97–0.997`
- `medium_launcher` and `large_tower` around frames 102–116:
  - often `0.85–0.98`
- `small_plane` / `jet_plane` / `hangar` around frames 119–135:
  - many detections `0.9–0.99`
- `tank` around frames 139–152:
  - approximately `0.83–0.99`
- `small_tower` around frames 166–180:
  - approximately `0.96–0.99`
- `tank` around frames 192–207:
  - approximately `0.85–0.95`
- `large_launcher` around frames 205–219:
  - approximately `0.96–0.999`

At the same time, many secondary detections around these tracks remained in the `0.020–0.050` range.

### Interpretation

The hidden-domain detector appears substantially better than the raw `0.001` output initially suggested.

Strong real-looking tracks are often extremely confident.

The main current failure mode is likely overly permissive post-processing rather than complete detector failure.

---

# EXP-D053E — Planned clean confidence ablation

Next planned experiment:

Change only:

`CONFIDENCE = 0.001 -> 0.030`

Keep fixed:

- D040 seed 21 + seed 7
- `MODEL_NMS_IOU = 0.70`
- `ENSEMBLE_NMS_IOU = 0.65`
- `IMAGE_SIZE = 960`
- stable centered L1 camera
- no validation capture
- no concurrent Medical training if possible

### Purpose

Measure the causal effect of global confidence filtering without confounding it with:

- NMS changes
- class suppression
- camera changes
- capture I/O
- synthetic specialist logic

### Follow-up plan

If `0.030` improves remote score:

1. test `0.050`
2. then tune ensemble NMS separately

If `0.030` is neutral:

1. test `0.020`

If `0.030` significantly worsens:

1. explore `0.010–0.020`

Class-specific thresholds should only be tested after establishing the best global confidence range.

---

## EXP-D054A — Quadrant-scanning camera policy

**Status:** REJECTED

### Goal

Test whether poor remote performance was primarily caused by limited camera coverage.

The stable D031 policy remained at centered L1 for nearly the whole sequence, covering only the central 1920x1080 source region.

D054A instead cycled:

L0 -> L1 quadrant -> L0 -> next L1 quadrant

using the four legal L1 centers:

- (960, 540)
- (2880, 540)
- (2880, 1620)
- (960, 1620)

The reset to L0 was used between quadrants to avoid L1->L1 movement races.

### Configuration

Detector unchanged from the old D041 baseline:

- D040 seed21 + seed7
- confidence: 0.001
- model NMS IoU: 0.70
- ensemble NMS IoU: 0.65
- imgsz: 960

### Remote result

- score: `0.1093645809`
- received: `247/249`
- missing: `2`
- median latency: `54.0 ms`
- max latency: `155.0 ms`
- errors: none

### Interpretation

Uniformly increasing spatial coverage strongly reduced score.

The experiment was technically healthy, so the degradation was caused by the policy itself rather than transport or camera-command errors.

Likely reasons:

- too much time spent in L0 reset frames
- reduced temporal coverage of useful centered-L1 targets
- many quadrant views contained less useful signal than the center
- coverage alone is not sufficient without temporal tracking

### Decision

Reject naive quadrant scanning.

Return to stable centered L1.

---

## EXP-D054B — Invalid confidence-threshold run

**Status:** INVALID / DO NOT INTERPRET

### Goal

Test `CONFIDENCE = 0.03`.

### Remote result

- score: `0.0960639318`
- received: `248/249`
- missing: `1`
- median latency: `45.8 ms`
- max latency: `181.7 ms`
- errors: none

### Problem

The D054 quadrant-scanning predictor was accidentally still active.

Final telemetry showed:

`level=0 center=(1920,1080)`

instead of the intended stable centered L1 policy.

### Decision

Do not use this score to judge confidence 0.03.

---

## EXP-D054C — Clean global confidence 0.03 baseline

**Status:** COMPLETED

### Goal

Measure the effect of raising detector confidence from 0.001 to 0.03 with all other variables restored to the stable baseline.

### Configuration

Camera:

- D031 stable centered L1
- center: `(1920,1080)`

Detector:

- D040 seed21 + seed7
- confidence: `0.03`
- model NMS IoU: `0.70`
- ensemble NMS IoU: `0.65`
- imgsz: `960`

No temporal tracker.

### Remote result

- score: `0.1585120505`
- received: `236/249`
- missing: `13`
- median latency: `43.2 ms`
- max latency: `140.3 ms`
- errors: none
- final camera: L1 `(1920,1080)`

### Interpretation

Raising the global threshold to 0.03 significantly reduced score versus the older ~0.175 baseline.

Therefore the low-confidence tail is not purely harmful; some low-confidence predictions improve AP ranking / recall.

### Decision

Do not globally raise production confidence to 0.03.

Use a lower detector threshold, but restrict which detections are allowed to enter temporal memory.

---

# EXP-D055 — Temporal object memory

## EXP-D055A-v1 — Initial temporal tracker

**Status:** REJECTED

### Goal

Add short-term temporal object memory on top of the stable centered-L1 detector.

### Design

- class-agnostic geometric association
- per-track velocity estimate
- confidence/class voting
- short missed-frame persistence
- current detections + memory detections
- report-level duplicate suppression

### Offline replay result

Across 230 captured validation frames:

- current detections: `532`
- memory detections: `482`
- final detections: `881`
- frames with memory: `135`
- avg current/frame: `2.313`
- avg memory/frame: `2.096`
- avg final/frame: `3.83`
- max memory/frame: `11`
- max final/frame: `19`

### Failure mode

The tracker accumulated ghost copies of the same object.

Example:

- one persistent tank detection gradually produced 4 extra remembered tank tracks

### Decision

Reject v1.

Add stale-track suppression and stricter memory qualification.

---

## EXP-D055A-v2 — Conservative temporal memory

**Status:** OFFLINE SUCCESS

### Changes

- require at least 3 hits before propagation
- require a strong observation >= 0.60
- max missed frames reduced to 3
- faster confidence decay
- kill unmatched tracks that overlap current detections
- internal track deduplication

### Offline replay result

Across 230 captured validation frames:

- current detections: `532`
- memory detections: `23`
- final detections before safe-merge fix: `431`
- frames with memory: `18`
- avg memory/frame: `0.10`
- max memory/frame: `3`

### New issue

Class-agnostic final NMS removed too many legitimate current detections, especially crowded aircraft frames.

### Decision

Preserve all current detections.

Allow memory only to supplement detector gaps.

---

## EXP-D055A-v3 — Safe supplemental memory

**Status:** OFFLINE SUCCESS

### Changes

Current detector predictions are never removed by tracker post-processing.

Memory is added only when it does not overlap an existing current detection.

### Offline replay result

Across 230 captured validation frames:

- current detections: `532`
- memory detections: `23`
- final detections: `555`
- frames with memory: `18`
- avg current/frame: `2.313`
- avg memory/frame: `0.100`
- avg final/frame: `2.413`
- max memory/frame: `3`
- max final/frame: `18`

Useful gap-filling examples included:

- frame 137: current 0 -> memory 3
- frames 181-183: current 0 -> memory 1
- frame 222: current 0 -> memory 1
- frame 224: current 0 -> memory 1

### Interpretation

Tracker now behaves as intended:

- small number of additions
- fills short detector gaps
- does not multiply persistent objects
- does not delete current detections

---

## EXP-D055B — Reproducibility confirmation

- Best score: `0.1802420600`
- Confirmation score: `0.1793168238`
- Confirmation telemetry: `241/249`, 8 missing, median `44.5 ms`, max `145.0 ms`
- Camera: stable L1 `(1920,1080)`
- Tracker input threshold: `>= 0.03`
- Decision: keep as stable known-good baseline.

---

## EXP-D055C — Longer temporal persistence

**Status:** REJECTED

Change:

```python
MAX_MISSED_FRAMES = 4
```

Remote results:

- Run 1: `0.1752208183`, `239/249`, 10 missing
- Run 2: `0.1779436307`, `242/249`, 7 missing

Interpretation: extending persistence from 3 to 4 frames kept stale boxes alive too long and slightly hurt precision.

Decision: restore `MAX_MISSED_FRAMES = 3`.

---

## EXP-D056A — Hard-negative/domain-adapted fine-tuning

**Status:** TRAINED / NOT SELECTED FOR DEPLOYMENT

### Goal

Fine-tune D040 seed21 using a mixed dataset containing existing positives, manually verified real-domain examples, rare-class examples, and validation-domain hard negatives.

### Dataset

```text
D032 positives:              77
D048 real-domain positives: 171
D045 selected positives:    600
hard-negative images:        66
Total:                      914
```

All 16 classes were represented. The 66 hard negatives had empty YOLO label files.

Class counts:

```text
0  condor             117
1  hangar              15
2  helicopter          29
3  jammer             110
4  jet_plane           21
5  large_launcher      15
6  large_tower         46
7  medium_launcher     12
8  medium_plane        18
9  mine_roller         21
10 small_launcher     140
11 small_plane         53
12 small_tower         39
13 spacecraft         108
14 ta-ta              125
15 tank                93
```

### Training

Base checkpoint: D040 seed21 best checkpoint.

Approximate configuration:

```text
epochs:        80
imgsz:         960
batch:         2
optimizer:     AdamW
lr0:           0.0005
cosine LR:     enabled
mosaic:        0.4
degrees:       8
translate:     0.08
scale:         0.25
fliplr:        0.5
flipud:        0.15
```

### Local validation

```text
precision:  0.518
recall:     0.712
mAP50:      0.637
mAP50-95:   0.498
```

Selected per-class mAP50:

```text
condor             0.995
hangar             0.828
helicopter         0.751
jammer             0.816
jet_plane          0.995
large_launcher     0.111
large_tower        0.694
medium_launcher    0.995
medium_plane       0.378
mine_roller        0.895
small_launcher     0.320
small_plane        0.635
small_tower        0.359
spacecraft         0.599
ta-ta              0.332
tank                0.492
```

Interpretation: several important validation classes remained weak, especially `large_launcher`, `medium_plane`, `small_tower`, and `tank`.

Decision: do not deploy D056A alone; test it conservatively as one ensemble member.

---

## EXP-D056B — D056A seed21 + original D040 seed7

**Status:** REJECTED

Configuration:

```text
seed21 -> D056A best.pt
seed7  -> original D040 seed7
CONFIDENCE = 0.001
MODEL_NMS_IOU = 0.70
ENSEMBLE_NMS_IOU = 0.65
IMAGE_SIZE = 960
D055B tracker
tracker input >= 0.03
MAX_MISSED_FRAMES = 3
stable centered L1
```

Remote result:

```text
score:             0.1731394852
received:          242/249
missing:           7
median latency:    44.4 ms
max latency:       165.5 ms
errors:            none
final camera:      L1 (1920,1080)
final detections:  85
```

Comparison:

```text
D055B best: 0.1802420600
D056B:      0.1731394852
delta:     -0.0071025748
```

Interpretation: technically healthy run but worse score; the 85 final-frame detections also suggested added noise.

Decision: reject D056B and restore the original D040 seed21 + seed7 ensemble.

---

## EXP-D057A — Domain-matched synthetic fine-tuning

**Status:** LOCALLY PROMISING / NOT FINAL

Reference data recovered:

- Helsinki images: 25
- Helsinki annotations: 25
- captured validation views: 230
- hard-negative pool: 66

Synthetic dataset:

```text
3000 synthetic images
171 D048 real-domain images
```

Balanced pasted-object counts were roughly 500–585 per class across all 16 classes.

Training from D040 seed21:

```text
Epoch 1: mAP50 0.633 | mAP50-95 0.441
Epoch 2: mAP50 0.792 | mAP50-95 0.584
Epoch 3: mAP50 0.906 | mAP50-95 0.675
```

Captured-validation comparison:

```text
threshold >= 0.03: D040 451, D057 632
threshold >= 0.10: D040 316, D057 255
threshold >= 0.30: D040 252, D057 101
```

D057 began predicting classes D040 rarely or never produced in the captured validation domain, but calibration shifted substantially.

---

## EXP-D057B — D057A seed21 + original D040 seed7

**Status:** REJECTED

Configuration:

```text
seed21: D057A
seed7: original D040 seed7
CONFIDENCE = 0.001
tracker input >= 0.10
stable centered L1
```

Remote result:

```text
score:             0.1712012003
received:          244/249
missing:           5
median latency:    47.5 ms
max latency:       145.1 ms
errors:            none
final detections:  63
```

Decision: reject D057B. Detector domain adaptation alone did not solve the remote bottleneck.

---

## EXP-D058A — Stale-safe active L1 camera sweep

**Status:** BEST SINGLE VALIDATION SCORE / HIGH VARIANCE

Kept the D040 ensemble and D055B tracker, but replaced the fixed camera with a stale-state-aware L1 tile pass followed by a continuous top-band L1 sweep. No L2 dives.

Remote results:

```text
Run 1: 0.1867719026 | 232/249 | 17 missing | 46.2 ms median
Run 2: 0.1684754147 | 222/249 | 27 missing | 44.9 ms median
Run 3: 0.1501361835 | 216/249 | 33 missing | 80.6 ms median
```

Run 1 became the highest validation score achieved during the competition.

Interpretation: active coverage helped, but performance was highly sensitive to skipped frames.

---

## EXP-D058B — Homography dead-reckoning + active L1 sweep

**Status:** REJECTED FOR FINAL DEPLOYMENT

Used the D058A camera with motion-aware global tracking based on the fitted homography:

```text
[ 1.006756  -0.001473  -12.614238 ]
[ 0.000337   1.011899   51.706644 ]
[ 0.0       -0.000001    1.0      ]
```

Remote results:

```text
Run 1: 0.1857364221 | 226/249 | 23 missing
Run 2: 0.1817151839 | 234/249 | 15 missing
```

Interpretation: functional, but no reproducible gain over simpler D055B memory.

Decision: restore D055B tracker.

---

## EXP-D059 — AWS Stockholm CPU deployment test

**Status:** REJECTED

AWS instance: `m7i-flex.large`, CPU-only.

Single-model CPU benchmark at imgsz 960:

```text
mean: 90.7 ms
min:  83.5 ms
max:  95.3 ms
```

Exact D055B runtime was deployed to AWS.

Remote result:

```text
score:             0.1561489533
received:          216/249
missing:           33
median latency:    200.9 ms
max latency:       227.9 ms
errors:            none
```

Decision: reject AWS CPU deployment and return to the Windows GTX 1060 endpoint.

---

## Final pre-evaluation Windows confirmation

Selected final runtime:

```text
D040 seed21 + seed7
CONFIDENCE = 0.001
MODEL_NMS_IOU = 0.70
ENSEMBLE_NMS_IOU = 0.65
D055B tracker
tracker input >= 0.03
EXP-D031 stable centered L1 camera
Windows GTX 1060
```

Final validation before hidden evaluation:

```text
score:          0.1779861215
received:       231/249
missing:        18
median latency: 47.1 ms
max latency:    142.4 ms
errors:         none
```

---

# FINAL HIDDEN EVALUATION

**Status:** COMPLETED

Runtime used:

```text
D040 seed21 + seed7 ensemble
CONFIDENCE = 0.001
D055B temporal tracker
tracker input >= 0.03
stable centered L1 camera
Windows GTX 1060 endpoint
```

Final result:

```text
score: 0.0181381435
errors: none
```

Comparison:

```text
best public validation: 0.1867719026  (D058A)
best stable D055B:      0.1802420600
final hidden evaluation:0.0181381435
```

Interpretation: the endpoint remained operational, so the severe drop indicates a major generalization failure rather than a server crash.

Likely contributors:

1. camera coverage overfitting from fixed centered L1
2. repeated validation-scene specialization
3. detector domain shift
4. choosing validation reproducibility over scene-independent coverage

Main lesson:

> Repeated validation can become a hidden form of training. A robust solution must optimize for scene-independent coverage, temporal geometry, and domain generalization rather than maximum performance on one repeatable validation sequence.

---

# Final experiment summary

```text
D055B stable temporal tracker:
best = 0.1802420600

D056 hard-negative/domain fine-tune:
remote ensemble = 0.1731394852

D057 synthetic-domain detector:
local synthetic mAP50 = 0.906 after 3 epochs
remote ensemble = 0.1712012003

D058A stale-safe active L1 sweep:
best = 0.1867719026

D058B homography tracker + L1 sweep:
best = 0.1857364221

AWS CPU deployment:
0.1561489533

Final hidden evaluation:
0.0181381435
```

## Best public validation configuration

```text
EXP-D058A
score = 0.1867719026
```

## Most reproducible public validation configuration

```text
EXP-D055B
~0.179–0.180
```

## Final hidden result

```text
0.0181381435
```

## Post-competition direction

If rebuilding without competition time pressure, prioritize:

1. scene-independent active camera coverage
2. motion-aware global tracking
3. online motion estimation instead of a fixed trajectory assumption
4. much larger domain-randomized synthetic training
5. strict separation between validation analysis and model-development data
6. validation across multiple independently generated scenes
7. confidence calibration on held-out domains
8. GPU hosting geographically close to the evaluator
9. camera-policy evaluation independent of detector training data
10. experiment selection based on cross-scene robustness rather than one repeatable validation sequence
