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

---