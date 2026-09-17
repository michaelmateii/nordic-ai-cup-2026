from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import librosa

import torch
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
)


DEFAULT_AUDIO_DIR = Path(
    r"C:\Users\Calle\Projects\Nordic-AI-Cup-2026-official"
    r"\medical-appointment\data\audio"
)

MODEL_ID = "distil-whisper/distil-medium.en"


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
        "--timestamps",
        choices=("word", "segment"),
        default="word",
        help="Timestamp granularity requested from the ASR pipeline.",
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_ID,
        help="Hugging Face Whisper model ID.",
    )

    return parser.parse_args()


def choose_audio(audio_dir: Path, requested: Path | None) -> Path:
    if requested is not None:
        audio_path = requested
    else:
        files = sorted(audio_dir.glob("*.mp3"))

        if not files:
            raise FileNotFoundError(
                f"No MP3 files found in: {audio_dir}"
            )

        audio_path = files[0]

    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    return audio_path


def main() -> None:
    args = parse_args()
    
    model_id = args.model

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch.")

    audio_path = choose_audio(
        args.audio_dir,
        args.audio,
    )

    print("=" * 72)
    print("Nordic AI Cup 2026 — Medical ASR probe")
    print("=" * 72)
    print(f"Model:       {model_id}")
    print(f"Audio:       {audio_path}")
    print(f"PyTorch:     {torch.__version__}")
    print(f"CUDA:        {torch.version.cuda}")
    print(f"GPU:         {torch.cuda.get_device_name(0)}")
    print(
        "GPU VRAM:    "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )
    print()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print("Loading processor...")
    load_start = time.perf_counter()

    processor = AutoProcessor.from_pretrained(model_id)

    print("Loading model...")

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )

    model.to("cuda")
    model.eval()

    asr = pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch.float16,
        device=0,
    )

    torch.cuda.synchronize()

    load_seconds = time.perf_counter() - load_start

    allocated_after_load = (
        torch.cuda.memory_allocated() / 1024**3
    )

    reserved_after_load = (
        torch.cuda.memory_reserved() / 1024**3
    )

    print()
    print(f"Model load:  {load_seconds:.2f} s")
    print(f"Allocated:   {allocated_after_load:.2f} GB")
    print(f"Reserved:    {reserved_after_load:.2f} GB")
    print()
    print("Transcribing...")
    
    print("Decoding audio with librosa...")

    audio_array, sample_rate = librosa.load(
        audio_path,
        sr=16000,
        mono=True,
    )

    audio_input = {
        "array": audio_array,
        "sampling_rate": sample_rate,
    }

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    inference_start = time.perf_counter()

    generate_kwargs = {}

    if not model_id.endswith(".en"):
        generate_kwargs = {
            "task": "transcribe",
            "language": "en",
        }
    
    timestamp_mode = (
        "word"
        if args.timestamps == "word"
        else True
    )

    result = asr(
        audio_input,
        return_timestamps=timestamp_mode,
        chunk_length_s=25,
        batch_size=1,
        generate_kwargs=generate_kwargs,
    )

    torch.cuda.synchronize()

    torch.cuda.synchronize()

    inference_seconds = (
        time.perf_counter() - inference_start
    )

    peak_allocated = (
        torch.cuda.max_memory_allocated() / 1024**3
    )

    peak_reserved = (
        torch.cuda.max_memory_reserved() / 1024**3
    )

    chunks = result.get("chunks", [])

    print()
    print("=" * 72)
    print("RESULT")
    print("=" * 72)
    print(f"Inference:    {inference_seconds:.2f} s")
    print(f"Peak alloc:   {peak_allocated:.2f} GB")
    print(f"Peak reserve: {peak_reserved:.2f} GB")
    print(f"Words/chunks: {len(chunks)}")
    print()

    print("TRANSCRIPT")
    print("-" * 72)
    print(result["text"].strip())
    print()

    print("FIRST 25 TIMESTAMPED CHUNKS")
    print("-" * 72)

    for item in chunks[:25]:
        timestamp = item.get("timestamp")
        text = item.get("text", "")

        print(
            f"{timestamp!s:>20}  {text!r}"
        )

    payload = {
        "model": model_id,
        "audio": str(audio_path),
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "peak_allocated_gb": peak_allocated,
        "peak_reserved_gb": peak_reserved,
        "text": result["text"],
        "chunks": chunks,
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
        print(f"Saved JSON:  {args.output}")


if __name__ == "__main__":
    main()