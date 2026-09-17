# Nordic AI Cup 2026 — Medical Appointment Experiment Log

Machine: Windows PC — NVIDIA GeForce GTX 1060 6 GB
Python: 3.12.10
PyTorch: 2.14.0+cu126
CUDA runtime reported by PyTorch: 12.6

All downstream validation must be conversation-disjoint where training or tuning is involved.

---

## EXP-M000 — Official Baseline and Evaluator Sanity Check

### Hypothesis

The official evaluator, supplied baseline, and oracle should reproduce the documented reference scores before any custom model development begins.

### Change made

None. Used the unmodified official Nordic AI Cup Medical Appointment implementation.

Official repository:

`C:\Users\Calle\Projects\Nordic-AI-Cup-2026-official\medical-appointment`

### Validation

Full supplied training set:

* 39 conversations
* 390 questions
* 195 positive
* 142 hard_negative
* 53 off_topic

### Oracle result

* Accuracy: 1.000
* mean tIoU: 1.000
* final score: 1.000

### Official baseline result

The supplied baseline answers TRUE to every question and provides no evidence span.

Results:

* Accuracy: 0.500
* positive accuracy: 1.000 (195/195)
* hard_negative accuracy: 0.000 (0/142)
* off_topic accuracy: 0.000 (0/53)
* mean tIoU: 0.000
* final score: 0.200
* mean round-trip latency: 82 ms/conversation
* worst round-trip latency: 2068 ms/conversation
* failed conversations: 0
* timeouts: 0

### Interpretation

The evaluator behaves consistently with the documented scoring formula.

Missing a gold-positive question costs both classification accuracy and its tIoU contribution.

The official baseline score of 0.200 and oracle score of 1.000 are reproduced.

### Decision

KEEP as permanent reference baseline.

---

## EXP-M001 — Distil-Whisper Large-v3 Transformers ASR

### Hypothesis

`distil-whisper/distil-large-v3` may provide accurate English medical transcription with word timestamps while fitting within GTX 1060 6 GB VRAM and the 60-second request limit.

### Change made

Created:

`medical/src/asr/probe_distil_whisper.py`

Pipeline:

MP3
→ librosa decode to 16 kHz mono
→ Distil-Whisper large-v3 on CUDA FP16
→ 25-second chunked Transformers ASR
→ word-level timestamps

Model:

`distil-whisper/distil-large-v3`

### Validation

Two supplied consultations were tested:

* `conversation_sample_10.mp3`
* `conversation_sample_20.mp3`

Sample 20 is the longest supplied training consultation.

### Results

#### conversation_sample_10

* inference latency: 23.02 s
* model allocated after load: 1.41 GB
* peak allocated VRAM: 4.59 GB
* peak reserved VRAM: 4.92 GB
* timestamped words/chunks: 288
* word timestamps: successful

Qualitative transcription preserved the important clinical facts.

One medication spelling error observed:

* expected entity approximately `esomeprazole`
* ASR produced `Esomeprosol`

#### conversation_sample_20

* inference latency: 50.79 s
* model allocated after load: 1.41 GB
* peak allocated VRAM: 4.62 GB
* peak reserved VRAM: 5.09 GB
* timestamped words/chunks: 658
* word timestamps: successful

Long-form chunk overlap produced duplicated transcript passages.

Examples included repetition around:

* stable pain assessment
* renewed prescription discussion

### Interpretation

Transcription quality and timestamp availability are promising.

However, 50.79 seconds of ASR alone on the longest observed training clip leaves only about nine seconds before the hard 60-second request timeout.

That margin is insufficient for:

* request decoding
* evidence retrieval
* yes/no classification
* evidence refinement
* API overhead
* latency variance
* potentially longer hidden conversations

Chunk-overlap duplication is also undesirable for evidence retrieval/localization.

### Decision

DISCARD as primary competition runtime configuration.

KEEP as a higher-quality ASR reference candidate for later comparison.

---

## EXP-M002 — Distil-Whisper Medium.en Transformers ASR

### Hypothesis

`distil-whisper/distil-medium.en` may substantially reduce latency and VRAM consumption while retaining sufficient clinical transcription quality and word timestamps.

### Change made

Changed ASR model to:

`distil-whisper/distil-medium.en`

### Initial setup result

Model loaded successfully on GTX 1060.

* allocated VRAM after model load: 0.74 GB
* reserved VRAM after model load: 0.75 GB

The first inference attempt did not run because the probe passed:

* `task="transcribe"`
* `language="en"`

to an English-only Whisper model.

Transformers correctly rejected those options.

### Retest 1

After removing the invalid `task` and `language` arguments for the English-only model, model loading again succeeded:

* model load: 3.86 s
* allocated VRAM after load: 0.74 GB
* reserved VRAM after load: 0.75 GB

Inference then failed before transcription because `distil-medium.en` does not provide `alignment_heads` in its generation configuration, which Transformers requires for token/word-level timestamps.

This is a timestamp-alignment compatibility limitation, not a CUDA, VRAM, or basic ASR failure.

### Interpretation

Do not manually copy alignment-head indices from a different Whisper architecture. The distilled model has a different decoder architecture, so doing so would create an unvalidated timestamping method.

The next controlled test will use segment-level timestamps only, solely to establish the model's latency and transcription quality.

If the model is not substantially faster than `distil-large-v3`, no further alignment work is justified.

### Decision

RETEST with segment-level timestamps for latency/quality measurement.

### Retest 2 — segment timestamps

The probe was modified to request segment-level timestamps instead of word-level timestamps.

Tested:

`conversation_sample_20.mp3`

Results:

* model load: 4.00 s
* inference latency: 29.21 s
* allocated VRAM after load: 0.74 GB
* reserved VRAM after load: 0.75 GB
* peak allocated VRAM: 0.84 GB
* peak reserved VRAM: 0.90 GB
* timestamped segments: 57
* segment timestamps: successful
* word timestamps: unavailable with this checkpoint/configuration

Clinical transcription quality remained broadly usable. Important concepts such as Pamol, ibumetin, paracetamol, widespread pain, stable pain, no examination, continuation of treatment, and renewal of both prescriptions were retained.

Chunk-overlap duplication remained present. One duplicated passage occurred around the pain-location discussion.

### Comparison with EXP-M001

`conversation_sample_20.mp3`:

* Distil-large-v3 + word timestamps: 50.79 s
* Distil-medium.en + segment timestamps: 29.21 s

Latency reduction:

approximately 42.5%

VRAM reduction was substantial:

* large-v3 peak allocated: 4.62 GB
* medium.en peak allocated: 0.84 GB

### Interpretation

`distil-medium.en` provides a substantially safer latency profile and leaves ample GPU memory for downstream components.

However, the checkpoint does not expose the alignment metadata required by Transformers for reliable word-level timestamps, and the challenge places 60% of the final score on temporal localization.

Segment spans are often several seconds long and therefore may impose a significant tIoU ceiling relative to the training annotations, whose median positive evidence duration is roughly 2.9 seconds.

The Transformers chunked pipeline also continues to create occasional duplicated passages.

### Decision

KEEP as a speed/quality reference.

DO NOT select as the final ASR yet.

Next experiment: evaluate faster-whisper/CTranslate2 for faster inference with native word timestamps.

---

## EXP-M003 — Whisper Small.en Transformers ASR

### Hypothesis

`openai/whisper-small.en` may provide a better latency/localization compromise than Distil-Whisper large-v3 because it is substantially smaller while retaining native Whisper alignment metadata required for word-level timestamps.

### Change made

Reuse the existing PyTorch/Transformers ASR probe with:

`openai/whisper-small.en`

Requested:

* CUDA FP16
* 25-second chunked inference
* word-level timestamps

### Validation

Primary latency test:

`conversation_sample_20.mp3`

This is the longest supplied training consultation.

### Results

Tested on:

`conversation_sample_20.mp3`

Results:

* inference latency: 57.86 s
* allocated VRAM after load: 0.47 GB
* reserved VRAM after load: 0.49 GB
* peak allocated VRAM: 2.10 GB
* peak reserved VRAM: 2.52 GB
* word timestamps: successful
* timestamped words/chunks: 660

The transcript retained the important clinical facts but still showed chunk-overlap duplication.

### Interpretation

Despite its smaller parameter count and lower memory usage, `openai/whisper-small.en` is slower than `distil-large-v3` on the GTX 1060 for this workload.

57.86 seconds of ASR on the longest supplied conversation leaves effectively no safe time budget for retrieval, classification, evidence refinement, and API overhead.

### Decision

DISCARD as primary runtime candidate.

---

## EXP-M004 — faster-whisper Distil-Large-v3 INT8/FP32

### Hypothesis

CTranslate2/faster-whisper may preserve the transcription quality and word-level timestamp capability of Distil-Whisper large-v3 while substantially reducing inference latency on the GTX 1060.

### Configuration

Backend:

`faster-whisper 1.2.1`

CTranslate2:

`4.8.2`

Model:

`distil-large-v3`

Device:

`cuda`

Compute type:

`int8_float32`

Beam size:

`1`

Other settings:

* word timestamps: enabled
* language: English
* VAD: disabled
* condition_on_previous_text: false

CTranslate2-reported CUDA compute types on GTX 1060:

* int8
* int8_float32
* float32

FP16 is not reported as a supported CUDA compute type on this machine.

### Validation

Primary test:

`conversation_sample_20.mp3`

### Initial runtime result

The faster-whisper model downloaded and loaded successfully, but first GPU inference failed with:

`RuntimeError: Library cublas64_12.dll is not found or cannot be loaded`

This confirms that the Python/CTranslate2 installation can detect the CUDA-capable GPU but the required external CUDA 12 cuBLAS runtime DLLs are not currently visible to Windows.

This is an environment/runtime dependency issue, not a model failure.

### Runtime fix

Project-local CUDA 12 cuBLAS and cuDNN 9 runtime DLLs were added to the Medical CMD session PATH.

No system-wide CUDA or cuDNN installation was required, and the working PyTorch CUDA environment was left unchanged.

### Successful CUDA retest

Configuration:

* backend: `faster-whisper 1.2.1`
* CTranslate2: `4.8.2`
* model: `distil-large-v3`
* device: `cuda`
* compute type: `int8_float32`
* beam size: `1`
* word timestamps: enabled
* language: English
* VAD: disabled
* `condition_on_previous_text`: false

Tested on:

`conversation_sample_20.mp3`

Results:

* model load: 4.30 s
* inference latency: 10.47 s
* segments: 64
* timestamped words: 607
* detected language: English
* language probability: 1.0000
* word timestamps: successful

Qualitative transcription preserved the major clinical facts, including:

* Pamol
* ibumetin / related medication references
* paracetamol
* generalized or widespread pain
* stable pain problem
* continued need for pain relief
* no examination performed during the visit
* no treatment change
* renewal of both prescriptions

Medication spelling varied between mentions, including forms such as `Ibumetan`, `ibupetin`, `Pamel`, and `Ibumetin`, but the underlying medication concepts were retained.

Unlike the Transformers chunked runs, no obvious duplicated overlap passages were observed.

### Comparison on conversation_sample_20.mp3

* Distil-large-v3 / Transformers / word timestamps: 50.79 s
* Distil-medium.en / Transformers / segment timestamps: 29.21 s
* Whisper-small.en / Transformers / word timestamps: 57.86 s
* Distil-large-v3 / faster-whisper / word timestamps: 10.47 s

The faster-whisper configuration is approximately 4.9x faster than the Transformers Distil-large-v3 run while retaining word-level timestamps.

### Interpretation

This is currently the strongest measured ASR configuration.

It combines:

* sufficiently strong qualitative transcription
* native word-level timestamps
* no observed chunk-overlap duplication
* substantial latency margin under the 60-second request timeout

A 10.47-second ASR time on the longest supplied training conversation leaves roughly 49 seconds for request decoding, evidence retrieval, classification, evidence refinement, response construction, and runtime variation.

The remaining ASR risk is primarily transcription accuracy for medications, numbers, doses, units, and short negations rather than latency.

### Decision

KEEP.

Current primary ASR candidate.

Next experiment: benchmark and cache all 39 supplied conversations using one persistent faster-whisper model instance.

---

## EXP-M005 — Full-Corpus faster-whisper Benchmark

### Hypothesis

The selected faster-whisper configuration should remain comfortably below the 60-second request timeout across the complete supplied training corpus when the model is loaded once and reused between conversations.

### Configuration

Backend:

`faster-whisper 1.2.1`

CTranslate2:

`4.8.2`

Model:

`distil-large-v3`

Device:

`cuda`

Compute type:

`int8_float32`

Beam size:

`1`

Other settings:

* word timestamps enabled
* language forced to English
* VAD disabled
* `condition_on_previous_text=false`
* one persistent loaded model reused across all conversations

### Validation

Complete supplied training corpus:

* conversations: 39
* one ASR pass per conversation
* timestamped transcripts cached locally

### Results

* mean inference latency: 5.53 s
* median inference latency: 4.98 s
* worst inference latency: 10.58 s
* best inference latency: 3.48 s
* total inference wall time: 215.90 s
* worst-latency file: `conversation_sample_20.mp3`

### Interpretation

The selected faster-whisper configuration provides a very large latency margin under the 60-second per-request timeout.

Even the slowest supplied conversation requires only 10.58 seconds for ASR, leaving roughly 49 seconds for:

* Base64/audio handling
* evidence retrieval
* yes/no classification
* hard-negative discrimination
* evidence refinement
* response validation
* runtime variance

Further ASR speed optimization currently has substantially lower expected value than improving question answering and evidence localization.

The cached timestamped transcripts now allow downstream retrieval/classification experiments to run without repeatedly paying ASR cost.

### Decision

KEEP.

Freeze `faster-whisper distil-large-v3 / CUDA int8_float32 / beam=1` as the primary ASR baseline until evidence shows that ASR accuracy is limiting downstream score.

Next experiment: evaluate timestamp-aware evidence retrieval against the 195 gold-positive evidence intervals.

