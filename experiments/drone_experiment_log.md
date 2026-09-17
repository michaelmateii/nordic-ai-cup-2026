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