from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SWEEP_DIR = Path(
    r"medical\artifacts\retrieval\overnight_m040_long"
)

SEED_FILES = {
    42: (
        SWEEP_DIR
        / "e4_lr2e5_base_seed42_all_candidates.csv"
    ),
    1337: (
        SWEEP_DIR
        / "e4_lr2e5_base_seed1337_all_candidates.csv"
    ),
    2026: (
        SWEEP_DIR
        / "e4_lr2e5_base_seed2026_all_candidates.csv"
    ),
    31415: (
        SWEEP_DIR
        / "e4_lr2e5_base_seed31415_all_candidates.csv"
    ),
}

CLASSIFICATION_PATH = Path(
    r"medical\artifacts\classification\nli_segmentwise_cv_results.csv"
)

OUTPUT = Path(
    r"medical\artifacts\evaluation\m041_seed_ensemble.csv"
)

METHOD = "max_segment_margin"


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()

    if text in {
        "true",
        "1",
        "yes",
    }:
        return True

    if text in {
        "false",
        "0",
        "no",
    }:
        return False

    raise ValueError(
        f"Cannot parse boolean: {value!r}"
    )


def load_seed(
    seed: int,
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing seed {seed} file: {path}"
        )

    df = pd.read_csv(path)

    required = {
        "question_id",
        "transcript_id",
        "candidate_index",
        "candidate_start",
        "candidate_end",
        "target_tiou",
        "oof_score",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            f"Seed {seed} missing columns: "
            f"{sorted(missing)}"
        )

    return df[
        [
            "question_id",
            "transcript_id",
            "candidate_index",
            "candidate_start",
            "candidate_end",
            "target_tiou",
            "oof_score",
        ]
    ].rename(
        columns={
            "oof_score": (
                f"score_{seed}"
            )
        }
    )


def add_percentile_rank(
    df: pd.DataFrame,
    score_column: str,
    output_column: str,
) -> None:
    df[output_column] = (
        df.groupby(
            "question_id"
        )[score_column]
        .rank(
            method="average",
            pct=True,
        )
    )


def select_best(
    df: pd.DataFrame,
    score_column: str,
) -> pd.DataFrame:
    indices = (
        df.groupby(
            "question_id"
        )[score_column]
        .idxmax()
    )

    return (
        df.loc[indices]
        .copy()
        .reset_index(drop=True)
    )


def summarize(
    name: str,
    selected: pd.DataFrame,
) -> dict[str, float]:
    scores = (
        selected["target_tiou"]
        .astype(float)
    )

    metrics = {
        "mean_tiou": float(
            scores.mean()
        ),
        "median_tiou": float(
            scores.median()
        ),
        "any_overlap": float(
            (scores > 0).mean()
        ),
        "tiou_ge_025": float(
            (scores >= 0.25).mean()
        ),
        "tiou_ge_050": float(
            (scores >= 0.50).mean()
        ),
        "tiou_ge_075": float(
            (scores >= 0.75).mean()
        ),
    }

    print()
    print(name)
    print("-" * 78)

    print(
        f"Mean tIoU:       "
        f"{metrics['mean_tiou']:.4f}"
    )

    print(
        f"Median tIoU:     "
        f"{metrics['median_tiou']:.4f}"
    )

    print(
        f"Any overlap:     "
        f"{metrics['any_overlap']:.4f}"
    )

    print(
        f"tIoU >= 0.25:    "
        f"{metrics['tiou_ge_025']:.4f}"
    )

    print(
        f"tIoU >= 0.50:    "
        f"{metrics['tiou_ge_050']:.4f}"
    )

    print(
        f"tIoU >= 0.75:    "
        f"{metrics['tiou_ge_075']:.4f}"
    )

    return metrics


def compute_composite(
    evidence: pd.DataFrame,
) -> tuple[
    float,
    float,
    float,
    int,
]:
    classification = pd.read_csv(
        CLASSIFICATION_PATH
    )

    classification = (
        classification[
            classification["method"]
            == METHOD
        ]
        .sort_values(
            "row_index"
        )
        .reset_index(drop=True)
    )

    if len(classification) != 390:
        raise RuntimeError(
            "Expected 390 classification rows; "
            f"got {len(classification)}"
        )

    classification["gold_bool"] = [
        parse_bool(value)
        for value
        in classification["gold_yes"]
    ]

    classification["pred_bool"] = [
        parse_bool(value)
        for value
        in classification[
            "predicted_yes"
        ]
    ]

    accuracy = float(
        (
            classification["gold_bool"]
            == classification["pred_bool"]
        ).mean()
    )

    positives = classification[
        classification["gold_bool"]
    ].copy()

    if len(positives) != 195:
        raise RuntimeError(
            "Expected 195 gold positives; "
            f"got {len(positives)}"
        )

    evidence_by_id = (
        evidence.set_index(
            "question_id"
        )
    )

    scored = []

    returned_count = 0

    for _, row in positives.iterrows():
        if not row["pred_bool"]:
            scored.append(0.0)
            continue

        question_id = (
            row["question_id"]
        )

        if (
            question_id
            not in evidence_by_id.index
        ):
            raise RuntimeError(
                "Missing evidence for "
                f"{question_id}"
            )

        returned_count += 1

        scored.append(
            float(
                evidence_by_id.loc[
                    question_id,
                    "target_tiou",
                ]
            )
        )

    mean_scored_tiou = float(
        np.mean(scored)
    )

    composite = (
        0.4 * accuracy
        + 0.6 * mean_scored_tiou
    )

    return (
        accuracy,
        mean_scored_tiou,
        composite,
        returned_count,
    )


def main() -> None:
    frames = {
        seed: load_seed(
            seed,
            path,
        )
        for seed, path
        in SEED_FILES.items()
    }

    seeds = list(
        SEED_FILES.keys()
    )

    base_seed = seeds[0]

    merged = frames[
        base_seed
    ].copy()

    for seed in seeds[1:]:
        other = frames[
            seed
        ]

        merged = merged.merge(
            other[
                [
                    "question_id",
                    "candidate_index",
                    f"score_{seed}",
                ]
            ],
            on=[
                "question_id",
                "candidate_index",
            ],
            how="inner",
            validate="one_to_one",
        )

    expected_length = len(
        frames[base_seed]
    )

    if len(merged) != expected_length:
        raise RuntimeError(
            "Candidate merge lost rows: "
            f"expected={expected_length}, "
            f"actual={len(merged)}"
        )

    score_columns = [
        f"score_{seed}"
        for seed in seeds
    ]

    # ---------------------------------------------------------
    # Primary ensemble:
    # arithmetic mean of all four seed scores.
    # ---------------------------------------------------------

    merged[
        "ensemble_mean_score"
    ] = (
        merged[
            score_columns
        ].mean(axis=1)
    )

    # ---------------------------------------------------------
    # Diagnostic ensemble:
    # within-question percentile-rank average.
    #
    # This removes any global score-calibration differences
    # between independently trained seeds.
    # ---------------------------------------------------------

    rank_columns = []

    for seed in seeds:
        rank_column = (
            f"rank_{seed}"
        )

        add_percentile_rank(
            merged,
            f"score_{seed}",
            rank_column,
        )

        rank_columns.append(
            rank_column
        )

    merged[
        "ensemble_rank_score"
    ] = (
        merged[
            rank_columns
        ].mean(axis=1)
    )

    # Median is another robust diagnostic.
    merged[
        "ensemble_median_score"
    ] = (
        merged[
            score_columns
        ].median(axis=1)
    )

    print("=" * 78)
    print(
        "Nordic AI Cup 2026 — "
        "M041 Four-Seed Evidence Ensemble"
    )
    print("=" * 78)

    print(
        f"Seeds: {seeds}"
    )

    print(
        f"Candidates: {len(merged)}"
    )

    print()

    individual_results = {}

    for seed in seeds:
        selected = select_best(
            merged,
            f"score_{seed}",
        )

        individual_results[
            seed
        ] = summarize(
            f"SEED {seed}",
            selected,
        )

    mean_selected = select_best(
        merged,
        "ensemble_mean_score",
    )

    mean_metrics = summarize(
        "EQUAL RAW-SCORE MEAN ENSEMBLE",
        mean_selected,
    )

    rank_selected = select_best(
        merged,
        "ensemble_rank_score",
    )

    rank_metrics = summarize(
        "EQUAL WITHIN-QUESTION RANK ENSEMBLE",
        rank_selected,
    )

    median_selected = select_best(
        merged,
        "ensemble_median_score",
    )

    median_metrics = summarize(
        "MEDIAN SCORE ENSEMBLE",
        median_selected,
    )

    # ---------------------------------------------------------
    # Use raw equal-mean as the pre-specified primary M041
    # system. Rank/median are diagnostics.
    # ---------------------------------------------------------

    accuracy, scored_tiou, composite, returned = (
        compute_composite(
            mean_selected
        )
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    mean_selected.to_csv(
        OUTPUT,
        index=False,
    )

    print()
    print("=" * 78)
    print("PRIMARY M041 COMPOSITE")
    print("=" * 78)

    print(
        "Evidence method:       "
        "equal raw-score mean, 4 seeds"
    )

    print(
        f"Classification accuracy: "
        f"{accuracy:.4f}"
    )

    print(
        f"Mean scored tIoU:        "
        f"{scored_tiou:.4f}"
    )

    print(
        f"Composite score:         "
        f"{composite:.4f}"
    )

    print(
        f"Positive predicted YES:  "
        f"{returned}/195"
    )

    print()
    print("REFERENCE")
    print("-" * 78)

    print(
        "M030 sentence evidence:     0.5089"
    )

    print(
        "M036 hybrid seed42:         0.5144"
    )

    print(
        "M040 best single seed2026:  0.5179"
    )

    print(
        "M037 composite:             0.6190"
    )

    print()
    print(
        f"Saved primary ensemble: {OUTPUT}"
    )


if __name__ == "__main__":
    main()