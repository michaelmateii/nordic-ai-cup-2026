import numpy as np
import pandas as pd

d = pd.read_csv(
    "medical/artifacts/classification/"
    "m070_task_specific_nli_oof.csv"
)

g = (
    d["gold_bool"]
    .astype(str)
    .str.lower()
    .isin(["true", "1", "yes"])
    .to_numpy()
)

e = (
    d["evidence_tiou"]
    .fillna(0.0)
    .to_numpy()
)

p55 = d["m055_probability"].to_numpy()
p70 = d["m070_max"].to_numpy()

best = None

for w in np.arange(0.0, 1.001, 0.01):
    s = (
        w * p55
        + (1.0 - w) * p70
    )

    vals = np.unique(s)

    thresholds = np.r_[
        vals[0] - 1e-9,
        (vals[:-1] + vals[1:]) / 2.0,
        vals[-1] + 1e-9,
    ]

    for t in thresholds:
        p = s >= t

        accuracy = np.mean(
            p == g
        )

        tiou = np.mean(
            np.where(
                p[g],
                e[g],
                0.0,
            )
        )

        composite = (
            0.4 * accuracy
            + 0.6 * tiou
        )

        result = (
            composite,
            w,
            t,
            accuracy,
            tiou,
            int(p.sum()),
        )

        if (
            best is None
            or result[0] > best[0]
        ):
            best = result


print(
    "BEST "
    "composite weight threshold "
    "accuracy tiou predicted_yes"
)
print(best)


# Current deployed M071 rule
w = 0.70
t = 0.3053

s = (
    w * p55
    + (1.0 - w) * p70
)

p = s >= t

accuracy = np.mean(
    p == g
)

tiou = np.mean(
    np.where(
        p[g],
        e[g],
        0.0,
    )
)

composite = (
    0.4 * accuracy
    + 0.6 * tiou
)

print(
    "M071 fixed",
    composite,
    w,
    t,
    accuracy,
    tiou,
    int(p.sum()),
)