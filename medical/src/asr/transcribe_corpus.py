from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import ctranslate2
from faster_whisper import WhisperModel


DEFAULT_AUDIO_DIR = Path(
    Path(__file__).resolve().parents[3].parent
    / "Nordic-AI-Cup-2026-official"
    / "medical-appointment"
    / "data"
    / "audio"
)

DEFAULT_OUTPUT_DIR = Path(
    r"medical\artifacts\asr\faster_distil_large_v3_int8f32_b1"
)

DEFAULT_MODEL = "distil-large-v3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=DEFAULT_AUDIO_DIR,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--compute-type",
        type=str,
        default="int8_float32",
    )

    parser.add_argument(
        "--beam-size",
        type=int,
        default=1,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    audio_files = sorted(args.audio_dir.glob("*.mp3"))

    if not audio_files:
        raise FileNotFoundError(
            f"No MP3 files found in {args.audio_dir}"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    supported = ctranslate2.get_supported_compute_types("cuda")

    if args.compute_type not in supported:
        raise RuntimeError(
            f"{args.compute_type!r} not supported. "
            f"Supported CUDA compute types: {supported}"
        )

    print("=" * 78)
    print("Nordic AI Cup 2026 — Medical full ASR benchmark")
    print("=" * 78)
    print(f"Model:        {args.model}")
    print(f"Compute type: {args.compute_type}")
    print(f"Beam size:    {args.beam_size}")
    print(f"Audio files:  {len(audio_files)}")
    print(f"Output:       {args.output_dir}")
    print()

    load_start = time.perf_counter()

    model = WhisperModel(
        args.model,
        device="cuda",
        compute_type=args.compute_type,
    )

    load_seconds = time.perf_counter() - load_start

    print(f"Model load:   {load_seconds:.2f} s")
    print()

    rows: list[dict[str, object]] = []
    inference_times: list[float] = []

    corpus_start = time.perf_counter()

    for index, audio_path in enumerate(audio_files, start=1):
        print(
            f"[{index:02d}/{len(audio_files):02d}] "
            f"{audio_path.name}"
        )

        inference_start = time.perf_counter()

        segment_generator, info = model.transcribe(
            str(audio_path),
            language="en",
            beam_size=args.beam_size,
            word_timestamps=True,
            vad_filter=False,
            condition_on_previous_text=False,
        )

        segments = list(segment_generator)

        inference_seconds = time.perf_counter() - inference_start
        inference_times.append(inference_seconds)

        segment_payload = []
        word_payload = []
        transcript_parts = []

        audio_end = 0.0

        for segment in segments:
            transcript_parts.append(segment.text)
            audio_end = max(audio_end, float(segment.end))

            segment_words = []

            if segment.words is not None:
                for word in segment.words:
                    word_item = {
                        "start": word.start,
                        "end": word.end,
                        "word": word.word,
                        "probability": word.probability,
                    }

                    segment_words.append(word_item)
                    word_payload.append(word_item)

            segment_payload.append(
                {
                    "id": segment.id,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "avg_logprob": segment.avg_logprob,
                    "no_speech_prob": segment.no_speech_prob,
                    "words": segment_words,
                }
            )

        transcript = "".join(transcript_parts).strip()

        realtime_factor = (
            inference_seconds / audio_end
            if audio_end > 0
            else None
        )

        payload = {
            "backend": "faster-whisper",
            "model": args.model,
            "device": "cuda",
            "compute_type": args.compute_type,
            "beam_size": args.beam_size,
            "audio": str(audio_path),
            "load_seconds": load_seconds,
            "inference_seconds": inference_seconds,
            "estimated_audio_seconds": audio_end,
            "realtime_factor": realtime_factor,
            "language": info.language,
            "language_probability": info.language_probability,
            "text": transcript,
            "segments": segment_payload,
            "words": word_payload,
        }

        output_json = args.output_dir / f"{audio_path.stem}.json"

        output_json.write_text(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        rows.append(
            {
                "audio": audio_path.name,
                "inference_seconds": round(inference_seconds, 4),
                "audio_seconds": round(audio_end, 4),
                "realtime_factor": (
                    round(realtime_factor, 5)
                    if realtime_factor is not None
                    else ""
                ),
                "segments": len(segment_payload),
                "words": len(word_payload),
                "language_probability": (
                    round(info.language_probability, 6)
                    if info.language_probability is not None
                    else ""
                ),
            }
        )

        print(
            f"    inference={inference_seconds:.2f}s "
            f"audio≈{audio_end:.2f}s "
            f"RTF={realtime_factor:.3f} "
            f"words={len(word_payload)}"
        )

    total_seconds = time.perf_counter() - corpus_start

    summary_csv = args.output_dir / "summary.csv"

    with summary_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"Conversations: {len(rows)}")
    print(
        f"Mean inference: "
        f"{statistics.mean(inference_times):.2f} s"
    )
    print(
        f"Median inference: "
        f"{statistics.median(inference_times):.2f} s"
    )
    print(
        f"Worst inference: "
        f"{max(inference_times):.2f} s"
    )
    print(
        f"Best inference: "
        f"{min(inference_times):.2f} s"
    )
    print(
        f"Total inference wall time: "
        f"{total_seconds:.2f} s"
    )

    worst_index = inference_times.index(max(inference_times))

    print(
        "Worst file:      "
        f"{audio_files[worst_index].name}"
    )

    print()
    print(f"Summary CSV:    {summary_csv}")


if __name__ == "__main__":
    main()