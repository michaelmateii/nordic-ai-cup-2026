from __future__ import annotations

import argparse
import json
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

DEFAULT_MODEL = "distil-large-v3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=DEFAULT_AUDIO_DIR,
    )

    parser.add_argument(
        "--audio",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
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


def choose_audio(
    audio_dir: Path,
    requested: Path | None,
) -> Path:
    if requested is not None:
        audio_path = requested
    else:
        files = sorted(audio_dir.glob("*.mp3"))

        if not files:
            raise FileNotFoundError(
                f"No MP3 files found in {audio_dir}"
            )

        audio_path = files[0]

    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    return audio_path


def main() -> None:
    args = parse_args()

    audio_path = choose_audio(
        args.audio_dir,
        args.audio,
    )

    supported = ctranslate2.get_supported_compute_types("cuda")

    print("=" * 72)
    print("Nordic AI Cup 2026 — faster-whisper ASR probe")
    print("=" * 72)
    print(f"Model:        {args.model}")
    print(f"Audio:        {audio_path}")
    print(f"Device:       cuda")
    print(f"Compute type: {args.compute_type}")
    print(f"Beam size:    {args.beam_size}")
    print(f"CT2 CUDA types: {supported}")
    print()

    if args.compute_type not in supported:
        raise RuntimeError(
            f"Compute type {args.compute_type!r} is not supported. "
            f"Supported: {supported}"
        )

    print("Loading model...")

    load_start = time.perf_counter()

    model = WhisperModel(
        args.model,
        device="cuda",
        compute_type=args.compute_type,
    )

    load_seconds = time.perf_counter() - load_start

    print(f"Model load:   {load_seconds:.2f} s")
    print()
    print("Transcribing...")

    inference_start = time.perf_counter()

    segments_generator, info = model.transcribe(
        str(audio_path),
        language="en",
        beam_size=args.beam_size,
        word_timestamps=True,
        vad_filter=False,
        condition_on_previous_text=False,
    )

    # IMPORTANT:
    # faster-whisper is lazy. Actual inference does not happen until
    # the generator is consumed.
    segments = list(segments_generator)

    inference_seconds = (
        time.perf_counter() - inference_start
    )

    segment_payload = []
    word_payload = []
    transcript_parts = []

    for segment in segments:
        transcript_parts.append(segment.text)

        segment_words = []

        if segment.words is not None:
            for word in segment.words:
                item = {
                    "start": word.start,
                    "end": word.end,
                    "word": word.word,
                    "probability": word.probability,
                }

                segment_words.append(item)
                word_payload.append(item)

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

    print()
    print("=" * 72)
    print("RESULT")
    print("=" * 72)
    print(f"Inference:    {inference_seconds:.2f} s")
    print(f"Segments:     {len(segment_payload)}")
    print(f"Words:        {len(word_payload)}")
    print(f"Language:     {info.language}")
    print(
        f"Language prob:{info.language_probability:.4f}"
        if info.language_probability is not None
        else "Language prob:n/a"
    )

    print()
    print("TRANSCRIPT")
    print("-" * 72)
    print(transcript)

    print()
    print("FIRST 25 TIMESTAMPED WORDS")
    print("-" * 72)

    for word in word_payload[:25]:
        print(
            f"({word['start']}, {word['end']}) "
            f"{word['word']!r} "
            f"p={word['probability']:.3f}"
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
        "language": info.language,
        "language_probability": info.language_probability,
        "text": transcript,
        "segments": segment_payload,
        "words": word_payload,
    }

    if args.output is not None:
        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        args.output.write_text(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print()
        print(f"Saved JSON:   {args.output}")


if __name__ == "__main__":
    main()