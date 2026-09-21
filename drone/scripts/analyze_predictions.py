"""
analyze_predictions.py
Quantitative summary of all captured validation prediction JSONs.
Outputs: per-class stats, confidence histograms, box-size distribution,
         frame-to-frame persistence, huge-box offenders, duplicate clusters.
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

# ── config ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[2]
CAPTURE_ROOT = ROOT / "drone" / "captures"

# Threshold candidates to evaluate offline
THRESHOLDS = [0.001, 0.005, 0.010, 0.020, 0.030, 0.050, 0.080, 0.100, 0.150, 0.200]

# Box area fraction above which a detection is "huge" (suspicious)
HUGE_BOX_FRACTION = 0.15   # >15% of image area

# Duplicate: two boxes of same class, IoU > this, same frame
DUPLICATE_IOU = 0.50

# Persistence: class appearing in same ~region across N+ consecutive frames
PERSIST_FRAMES = 5
PERSIST_DIST_PX = 80   # center distance in global coords

# ── helpers ───────────────────────────────────────────────────────────────────

def iou(a, b):
    """Both boxes are (x1,y1,x2,y2) in any consistent coordinate space."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (area_a + area_b - inter + 1e-9)

def box_area_fraction(ann, ow, oh):
    x1, y1, x2, y2 = (ann["bbox"][0]*ow, ann["bbox"][1]*oh,
                       ann["bbox"][2]*ow, ann["bbox"][3]*oh)
    return ((x2-x1) * (y2-y1)) / (ow * oh + 1e-9)

def box_center(ann, ow, oh):
    x1, y1, x2, y2 = (ann["bbox"][0]*ow, ann["bbox"][1]*oh,
                       ann["bbox"][2]*ow, ann["bbox"][3]*oh)
    return ((x1+x2)/2, (y1+y2)/2)

def conf_bucket(c):
    edges = [0.005, 0.010, 0.020, 0.030, 0.050, 0.080, 0.100, 0.150, 0.200, 0.300, 0.500, 1.001]
    for e in edges:
        if c < e:
            return f"<{e:.3f}"
    return ">=0.500"

# ── load data ─────────────────────────────────────────────────────────────────

sequences = [p for p in CAPTURE_ROOT.iterdir()
             if p.is_dir() and (p / "predictions").exists()]
if not sequences:
    sys.exit("No captured sequences found.")
SEQ = max(sequences, key=lambda p: p.stat().st_mtime)
PREDS = SEQ / "predictions"

pred_files = sorted(PREDS.glob("*.json"))
print(f"Sequence : {SEQ.name}")
print(f"Frames   : {len(pred_files)}\n")

all_anns = []   # flat list of (frame_idx, ann, ow, oh)
frames   = []   # list of (stem, [ann,...], ow, oh)

for fidx, pf in enumerate(pred_files):
    data = json.loads(pf.read_text(encoding="utf-8"))
    ow = data.get("original_width",  3840)
    oh = data.get("original_height", 2160)
    anns = data.get("annotations", [])
    frames.append((pf.stem, anns, ow, oh))
    for a in anns:
        all_anns.append((fidx, a, ow, oh))

# ── 1. threshold vs count table ───────────────────────────────────────────────

print("=" * 55)
print("1. DETECTIONS vs THRESHOLD")
print("=" * 55)
print(f"{'Threshold':>10}  {'Total':>8}  {'Per-frame':>10}  {'Dropped vs prev':>16}")
prev = None
for t in THRESHOLDS:
    kept = sum(1 for _, a, _, _ in all_anns if float(a["confidence"]) >= t)
    per  = kept / len(frames) if frames else 0
    drop = f"-{prev-kept} ({100*(prev-kept)/prev:.0f}%)" if prev is not None and prev>0 else ""
    print(f"{t:>10.3f}  {kept:>8}  {per:>10.1f}  {drop:>16}")
    prev = kept
print()

# ── 2. per-class breakdown ────────────────────────────────────────────────────

class_stats = defaultdict(lambda: {
    "count": 0,
    "confs": [],
    "area_fracs": [],
    "frames_seen": set(),
})

for fidx, ann, ow, oh in all_anns:
    cls = ann["object_id"]
    c   = float(ann["confidence"])
    af  = box_area_fraction(ann, ow, oh)
    class_stats[cls]["count"] += 1
    class_stats[cls]["confs"].append(c)
    class_stats[cls]["area_fracs"].append(af)
    class_stats[cls]["frames_seen"].add(fidx)

print("=" * 90)
print("2. PER-CLASS BREAKDOWN  (all detections, threshold=0.001)")
print("=" * 90)
header = f"{'Class':<22} {'Count':>7} {'Frames':>7} {'MeanConf':>9} {'P50Conf':>8} {'P95Conf':>8} {'MeanArea%':>10} {'MaxArea%':>9}"
print(header)
print("-" * 90)

def pct(lst, p):
    s = sorted(lst)
    idx = int(len(s) * p / 100)
    return s[min(idx, len(s)-1)]

for cls in sorted(class_stats, key=lambda c: -class_stats[c]["count"]):
    st = class_stats[cls]
    n  = st["count"]
    fs = len(st["frames_seen"])
    cs = st["confs"]
    af = st["area_fracs"]
    print(f"{cls:<22} {n:>7} {fs:>7} {sum(cs)/len(cs):>9.4f} "
          f"{pct(cs,50):>8.4f} {pct(cs,95):>8.4f} "
          f"{100*sum(af)/len(af):>10.2f} {100*max(af):>9.2f}")
print()

# ── 3. confidence histogram per class ────────────────────────────────────────

print("=" * 55)
print("3. CONFIDENCE HISTOGRAM  (top 8 classes by count)")
print("=" * 55)
top_classes = sorted(class_stats, key=lambda c: -class_stats[c]["count"])[:8]
buckets = ["<0.005","<0.010","<0.020","<0.030","<0.050","<0.080",
           "<0.100","<0.150","<0.200","<0.300","<0.500",">=0.500"]
hdr = f"{'Class':<22} " + "  ".join(f"{b:>7}" for b in buckets)
print(hdr)
print("-" * len(hdr))
for cls in top_classes:
    bc = defaultdict(int)
    for c in class_stats[cls]["confs"]:
        bc[conf_bucket(c)] += 1
    row = f"{cls:<22} " + "  ".join(f"{bc[b]:>7}" for b in buckets)
    print(row)
print()

# ── 4. huge-box offenders ────────────────────────────────────────────────────

print("=" * 55)
print(f"4. HUGE BOX OFFENDERS  (area > {HUGE_BOX_FRACTION*100:.0f}% of image)")
print("=" * 55)
huge_by_class = defaultdict(list)
for fidx, ann, ow, oh in all_anns:
    af = box_area_fraction(ann, ow, oh)
    if af >= HUGE_BOX_FRACTION:
        huge_by_class[ann["object_id"]].append((fidx, af, float(ann["confidence"])))

if not huge_by_class:
    print("  None found.\n")
else:
    for cls in sorted(huge_by_class, key=lambda c: -len(huge_by_class[c])):
        entries = huge_by_class[cls]
        areas   = [e[1] for e in entries]
        confs   = [e[2] for e in entries]
        fids    = sorted(set(e[0] for e in entries))
        print(f"  {cls:<22}  count={len(entries):>4}  "
              f"maxArea={100*max(areas):.1f}%  "
              f"meanConf={sum(confs)/len(confs):.4f}  "
              f"frames={len(fids)} ({fids[0]}..{fids[-1]})")
    print()

# ── 5. duplicate detection within frame ──────────────────────────────────────

print("=" * 55)
print(f"5. INTRA-FRAME DUPLICATES  (same class, IoU>{DUPLICATE_IOU})")
print("=" * 55)
dup_counts = defaultdict(int)
total_dups = 0

for stem, anns, ow, oh in frames:
    by_class = defaultdict(list)
    for a in anns:
        by_class[a["object_id"]].append(a)
    for cls, cls_anns in by_class.items():
        boxes = [(a["bbox"][0]*ow, a["bbox"][1]*oh,
                  a["bbox"][2]*ow, a["bbox"][3]*oh) for a in cls_anns]
        for i in range(len(boxes)):
            for j in range(i+1, len(boxes)):
                if iou(boxes[i], boxes[j]) >= DUPLICATE_IOU:
                    dup_counts[cls] += 1
                    total_dups += 1

print(f"  Total duplicate pairs: {total_dups}")
for cls in sorted(dup_counts, key=lambda c: -dup_counts[c]):
    print(f"  {cls:<22}  {dup_counts[cls]:>5} duplicate pairs")
print()

# ── 6. temporal persistence (likely hallucinations) ──────────────────────────

print("=" * 55)
print(f"6. TEMPORAL PERSISTENCE  (same class, >{PERSIST_FRAMES} consecutive frames, center<{PERSIST_DIST_PX}px)")
print("=" * 55)

# Build per-class frame->centers map
class_frame_centers = defaultdict(lambda: defaultdict(list))
for fidx, ann, ow, oh in all_anns:
    cx, cy = box_center(ann, ow, oh)
    class_frame_centers[ann["object_id"]][fidx].append((cx, cy, float(ann["confidence"])))

persistent_spots = []

for cls, frame_centers in class_frame_centers.items():
    fids = sorted(frame_centers.keys())
    if len(fids) < PERSIST_FRAMES:
        continue
    # sliding window: look for a cluster that survives PERSIST_FRAMES+ consecutive frames
    for start_i in range(len(fids)):
        run_len = 1
        for j in range(start_i+1, len(fids)):
            if fids[j] - fids[j-1] > 3:
                break
            # check if any center in fids[j] is within PERSIST_DIST_PX of any in fids[start_i]
            anchors = frame_centers[fids[start_i]]
            candidates = frame_centers[fids[j]]
            found = False
            for ax, ay, _ in anchors:
                for bx, by, _ in candidates:
                    if math.hypot(ax-bx, ay-by) <= PERSIST_DIST_PX:
                        found = True
                        break
                if found:
                    break
            if found:
                run_len += 1
            else:
                break
        if run_len >= PERSIST_FRAMES:
            confs_in_run = []
            for k in range(start_i, start_i+run_len):
                if k < len(fids):
                    for _, _, conf in frame_centers[fids[k]]:
                        confs_in_run.append(conf)
            persistent_spots.append((cls, fids[start_i], fids[start_i+run_len-1],
                                      run_len, sum(confs_in_run)/len(confs_in_run) if confs_in_run else 0))

# Sort by run length desc
persistent_spots.sort(key=lambda x: -x[3])

if not persistent_spots:
    print("  None found.\n")
else:
    print(f"  {'Class':<22} {'StartF':>7} {'EndF':>7} {'RunLen':>7} {'MeanConf':>9}")
    print(f"  {'-'*22} {'-'*7} {'-'*7} {'-'*7} {'-'*9}")
    shown = set()
    for cls, sf, ef, rl, mc in persistent_spots[:40]:
        key = (cls, sf)
        if key in shown:
            continue
        shown.add(key)
        print(f"  {cls:<22} {sf:>7} {ef:>7} {rl:>7} {mc:>9.4f}")
    print()

# ── 7. spatial distribution of FPs ──────────────────────────────────────────

print("=" * 55)
print("7. SPATIAL HOT-SPOTS  (where do boxes cluster?)")
print("=" * 55)
# Divide global image into 4x4 grid, count detections per cell
GRID_W = 4; GRID_H = 4

grid_counts = defaultdict(int)
for fidx, ann, ow, oh in all_anns:
    cx, cy = box_center(ann, ow, oh)
    gx = min(int(cx / ow * GRID_W), GRID_W-1)
    gy = min(int(cy / oh * GRID_H), GRID_H-1)
    grid_counts[(gx, gy)] += 1

print("  Detection count in 4x4 grid (x=left..right, y=top..bottom):")
print("       " + "  ".join(f"x={x}" for x in range(GRID_W)))
for gy in range(GRID_H):
    row = f"  y={gy}  " + "  ".join(f"{grid_counts[(gx,gy)]:>4}" for gx in range(GRID_W))
    print(row)
print()

# ── 8. recommended threshold summary ────────────────────────────────────────

print("=" * 55)
print("8. RECOMMENDED EXPERIMENT THRESHOLDS")
print("=" * 55)
for t in [0.020, 0.030, 0.050, 0.080, 0.100]:
    kept = sum(1 for _, a, _, _ in all_anns if float(a["confidence"]) >= t)
    per  = kept / len(frames) if frames else 0
    print(f"  conf>={t:.3f}  →  {kept:>5} total  ({per:.1f}/frame)")
print()
print("Recommendation: run official submissions at 0.030, 0.050, 0.080")
print("in that order. If 0.030 outperforms baseline, go lower next.")
print("If 0.080+ still beats baseline, your model recall is stronger")
print("than you think and you should push to 0.15.")
print()
print("Done.")