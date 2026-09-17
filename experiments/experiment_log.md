# Nordic AI Cup 2026 — Experiment Log
## EXP-S000 — Official random baseline

**Challenge:** Survival Simulator

**Hypothesis:** Official random controller provides a reference baseline.

**Validation:** Fixed 10-seed benchmark:
101, 202, 303, 404, 505, 606, 707, 808, 909, 1010

**Results:**
- Mean score: 35.0081
- Median score: 21.5951
- Min score: 14.7214
- Max score: 146.9914
- Stddev: 40.0914
- Mean survival: 34.39 s
- Median survival: 21.50 s

**Interpretation:** Very high variance. One outlier dominates the mean.

**Decision:** KEEP AS BASELINE

---

## EXP-S001 — Energy-aware straight-line controller

**Challenge:** Survival Simulator

**Hypothesis:** Removing random sprinting, random turning, and constant reproduction will improve robustness.

**Change made:**
- normal-speed movement only
- no random turning
- reproduction only above conservative energy threshold

**Validation:** Same fixed 10 seeds as EXP-S000.

**Results:**
- Mean score: 34.3445
- Median score: 32.1549
- Min score: 25.3000
- Max score: 45.3200
- Stddev: 6.7440
- Mean survival: 34.27 s
- Median survival: 32.10 s

**Interpretation:** Much more stable and much better median/worst-case performance, but mean did not improve due to loss of lucky exploratory behavior.

**Decision:** KEEP energy discipline; replace straight-line movement with smarter navigation.

---

## EXP-S002 — Fruit-seeking controller

**Challenge:** Survival Simulator

**Hypothesis:** Directly seeking detected fruit while retaining S001's energy discipline will substantially improve survival.

**Change made:**
- seek nearest detected fruit
- walk at normal speed
- occasionally turn when no fruit is detected
- retain conservative reproduction threshold
- no predator avoidance yet

**Validation:** Fixed seeds:
101, 202, 303, 404, 505, 606, 707, 808, 909, 1010

**Results:**
- Mean score: 378.4348
- Median score: 397.5066
- Min score: 42.9172
- Max score: 502.1767
- Stddev: 126.2072
- Mean survival: 381.26 s
- Median survival: 409.45 s
- Min survival: 42.70 s
- Max survival: 491.40 s

**Interpretation:** Fruit seeking produces a very large improvement over S001. Nine of ten seeds survive for hundreds of seconds, but seed 303 remains a catastrophic failure case.

**Decision:** KEEP. Use as the new controller foundation. Investigate predator avoidance and seed-303 failure next.

---

## EXP-S003 — Predator-aware fruit controller

**Challenge:** Survival Simulator

**Hypothesis:** Prioritizing predator avoidance while retaining S002 fruit seeking will improve survival and score.

**Change made:**
- retained nearest-fruit seeking
- flee from nearest detected predator
- sprint only when predator distance < 100
- retain conservative reproduction threshold

**Validation:** Fixed seeds:
101, 202, 303, 404, 505, 606, 707, 808, 909, 1010

**Results:**
- Mean score: 575.9413
- Median score: 681.3161
- Min score: 42.9172
- Max score: 847.4516
- Stddev: 271.4730
- Mean survival: 548.17 s
- Median survival: 657.50 s
- Min survival: 42.70 s
- Max survival: 805.30 s

**Interpretation:** Predator avoidance produces a major overall improvement over S002. Seed 303 is unchanged, indicating its failure is probably unrelated to predators. Seed 404 regresses substantially, suggesting the current flee/sprint strategy can overreact or interfere with food acquisition.

**Decision:** KEEP as new overall baseline. Investigate seed 303 separately and refine predator response.

## EXP-S004 — Stuck-aware fruit navigation

**Challenge:** Survival Simulator

**Hypothesis:** Some catastrophic failures are caused by agents repeatedly targeting food they are not actually making progress toward. Detecting lack of progress and temporarily escaping the trajectory should improve robustness.

**Change made:**
- retained S003 fruit seeking and predator avoidance
- added per-agent memory
- tracked distance to the current nearest fruit
- if progress remained below threshold for multiple steps:
  - marked the agent as stuck
  - temporarily moved perpendicular to the current trajectory
  - used a short escape period before resuming food seeking
- increased exploration strength when no fruit was detected
- retained conservative reproduction logic from S003

**Validation:** Fixed 10-seed development benchmark:
101, 202, 303, 404, 505, 606, 707, 808, 909, 1010

**Diagnostic validation:**
- Seed 303: 42.9172 → 805.0377
- Seed 404: 223.2728 → 858.2666

**Results:**
- Mean score: 752.7201
- Median score: 756.2087
- Min score: 417.3125
- Max score: 963.6791
- Stddev: 147.0732
- Mean survival: 727.39 s
- Median survival: 731.85 s
- Min survival: 403.70 s
- Max survival: 910.20 s

**Interpretation:** Stuck detection solved the catastrophic seed-303 starvation failure and substantially improved overall robustness. The failure was caused by navigation/food access rather than predators.

**Decision:** KEEP. Became the new controller foundation.

---

## EXP-S005 — Survival-first lineage strategy

**Challenge:** Survival Simulator

**Hypothesis:** Since the main score is dominated by keeping the species alive, reproduction should be used primarily to preserve the lineage rather than to maintain a large population.

**Change made:**
- retained S004 stuck-aware navigation and predator avoidance
- replaced simple high-energy reproduction with survival-focused reproduction
- reproduce when:
  - the agent is approaching old age and has sufficient reserve energy, or
  - the agent has a very large energy surplus
- reduced unnecessary reproduction energy expenditure

**Validation:** Fixed 10-seed development benchmark:
101, 202, 303, 404, 505, 606, 707, 808, 909, 1010

**Results:**
- Mean score: 847.4411
- Median score: 868.9837
- Min score: 494.4615
- Max score: 1097.4528
- Stddev: 180.3039
- Mean survival: 803.97 s
- Median survival: 820.05 s
- Min survival: 471.70 s
- Max survival: 1047.90 s

**Interpretation:** Survival-focused reproduction improved mean, median, worst-case score, and survival duration compared with S004. Several seeds exceeded 900 simulated seconds and seed 606 exceeded 1000 seconds.

**Decision:** KEEP. Became the strongest overall baseline before predator-behaviour exploitation.

---

## EXP-S006 — Predator-facing escape strategy

**Challenge:** Survival Simulator

**Hypothesis:** Because movement direction and facing direction are independent, agents can move away from predators while facing them, encouraging distant predators to pivot rather than directly chase.

**Change made:**
- retained S005 lineage, food seeking, and stuck recovery
- when predator distance > 95:
  - walk away at normal speed
  - face toward predator
- when predator distance <= 95:
  - sprint away
  - continue trying to keep predator in front

**Validation:** Same fixed 10 development seeds.

**Results:**
- Mean score: 884.2480
- Median score: 857.9500
- Min score: 564.2998
- Max score: 1460.1854
- Stddev: 252.6331
- Mean survival: 845.97 s
- Median survival: 800.95 s
- Min survival: 545.10 s
- Max survival: 1389.50 s

**Interpretation:** Predator-facing behavior is beneficial in some environments and increases mean score and worst-case score, but several seeds regress substantially. The mechanic appears exploitable, but the strategy needs conditional use rather than always taking priority over food acquisition.

**Decision:** KEEP AS CANDIDATE. Do not replace S005 as final policy yet.

---

## EXP-S007 — Conditional predator manipulation

**Challenge:** Survival Simulator

**Hypothesis:** Predator-facing behavior is useful only under favorable conditions. Immediate threats should trigger direct escape, moderate threats can use the stare/pivot mechanic when energy is healthy, and distant predators should be ignored in favor of food acquisition.

**Change made:**
- started from S005
- immediate predator danger (`<= 95`) triggers sprint escape
- moderate predator distance (`<= 180`) uses predator-facing manipulation only when energy is at least 55% of max
- distant predators are ignored
- food seeking and stuck-aware navigation remain the default behavior
- retained S005 survival-first reproduction

**Validation:** Fixed 10-seed development benchmark:
101, 202, 303, 404, 505, 606, 707, 808, 909, 1010

**Results:**
- Mean score: 1018.5050
- Median score: 1003.6121
- Min score: 732.2653
- Max score: 1322.0793
- Stddev: 165.0793
- Mean survival: 993.61 s
- Median survival: 982.45 s
- Min survival: 706.20 s
- Max survival: 1272.60 s

**Interpretation:** Conditional predator manipulation substantially outperforms both S005 and S006. It preserves robust food acquisition while capturing some of the predator-facing mechanic's upside.

**Decision:** KEEP. New primary survival controller.

---