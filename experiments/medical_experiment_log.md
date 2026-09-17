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

---

## EXP-M006 — TF-IDF Segment Evidence Retrieval

### Hypothesis

A simple lexical retrieval baseline over timestamped faster-whisper segments should establish how much positive-evidence localization is possible without semantic reranking or word-level span refinement.

### Configuration

Input:

* cached faster-whisper transcripts from EXP-M005
* only the 195 gold-positive training questions

Candidate units:

* faster-whisper ASR segments

Retrieval:

* word TF-IDF with 1–2 grams
* character TF-IDF with 3–5 grams
* combined score:

  * 0.55 word similarity
  * 0.45 character similarity

No question-answer classification was performed.

### Results

Gold-positive questions:

195

Top-1 localization:

* mean tIoU: 0.4016
* median tIoU: 0.4029
* any-overlap rate: 0.6718
* tIoU >= 0.25: 0.5897
* tIoU >= 0.50: 0.3949
* tIoU >= 0.75: 0.2615

Retrieval recall:

* R@1 any overlap: 0.6718
* R@3 any overlap: 0.8513
* R@5 any overlap: 0.8974

Segmentation ceiling:

* oracle single-segment mean tIoU: 0.6117
* oracle single-segment median tIoU: 0.5777

Failures:

* top-1 zero-overlap failures: 64 / 195

### Interpretation

The lexical baseline is useful but insufficient.

The increase from 67.2% overlap at rank 1 to 85.1% at rank 3 indicates that candidate generation is substantially stronger than top-1 ranking. A reranker therefore has meaningful potential.

However, even perfect selection among existing ASR segments is limited to only 0.6117 mean tIoU. Whole-segment output therefore imposes a major localization ceiling.

Because evidence localization represents 60% of the competition score, evidence spans must eventually be refined below ASR-segment granularity.

### Decision

KEEP as the first evidence-retrieval baseline.

Next experiment: quantify the oracle localization ceiling using faster-whisper word timestamps.

---

## EXP-M007 — Word Timestamp Localization Ceiling

### Hypothesis

If faster-whisper word timestamps are sufficiently accurate, contiguous word-level evidence windows should permit substantially higher temporal IoU than whole ASR segments.

### Method

Oracle diagnostic over the 195 gold-positive training questions.

For each question:

* load the cached faster-whisper word timestamps
* search contiguous word windows near the annotated evidence interval
* select the word-bounded span with maximum temporal IoU against the gold evidence

Gold evidence timestamps are used only for this diagnostic and are not available to or usable by a deployable inference system.

### Results

* gold-positive questions: 195
* mean oracle word-window tIoU: 0.9294
* median oracle word-window tIoU: 0.9457
* minimum oracle word-window tIoU: 0.6316
* tIoU >= 0.50: 1.0000
* tIoU >= 0.75: 0.9897
* tIoU >= 0.90: 0.8051

Previous oracle single-segment mean tIoU:

0.6117

Improvement from word-level boundaries:

+0.3177 mean tIoU

### Interpretation

faster-whisper's word timestamps are sufficiently accurate for high-quality evidence localization.

ASR temporal alignment is therefore not currently the dominant localization bottleneck.

The deployable challenge is selecting the correct contiguous word span from the transcript.

Whole ASR segments should not be used as final evidence spans except as fallback behavior.

### Decision

KEEP.

Use word-level evidence spans in the competition system.

Next experiment: lexical retrieval directly over candidate word windows without using gold evidence.


---

## EXP-M008 — Global Word-Window Evidence Retrieval

### Hypothesis

Directly ranking fixed-size contiguous word windows against each positive question may improve localization over whole-segment retrieval while retaining word-level timestamp precision.

### Configuration

Candidate word-window sizes:

* 6 words
* 10 words
* 14 words
* 18 words
* 24 words
* 32 words

Stride:

2 words

Ranking:

* word TF-IDF 1–2 grams
* character TF-IDF 3–5 grams
* 0.55 word similarity
* 0.45 character similarity

Evaluation used only the 195 gold-positive questions.

### Results

Top-1 localization:

* mean tIoU: 0.3522
* median tIoU: 0.3600
* any overlap: 0.7128
* tIoU >= 0.25: 0.6154
* tIoU >= 0.50: 0.3282
* tIoU >= 0.75: 0.1077

Retrieval recall:

* R@1 overlap: 0.7128
* R@3 overlap: 0.7744
* R@5 overlap: 0.8051
* R@10 overlap: 0.8615

Candidate-window ceiling:

* oracle mean tIoU: 0.7609
* oracle median tIoU: 0.7950

Top-1 zero-overlap failures:

56 / 195

Previous references:

* segment TF-IDF mean tIoU: 0.4016
* segment TF-IDF R@3: 0.8513
* unrestricted word timestamp oracle mean tIoU: 0.9294

### Interpretation

Global fixed-size word-window retrieval increases top-1 overlap frequency but reduces mean temporal IoU relative to whole-segment retrieval.

The fixed candidate set also captures only 0.7609 mean oracle tIoU, far below the 0.9294 unrestricted word-boundary ceiling.

This indicates that word timestamps are valuable for local span refinement, but global word windows are not an effective first-stage retrieval unit.

Segment retrieval remains the stronger coarse locator.

### Decision

DISCARD as standalone retrieval architecture.

KEEP the insight that evidence spans must be refined at word level.

Next experiment: coarse segment retrieval followed by local word-level refinement.

---

## EXP-M009 — Coarse-to-Fine TF-IDF Retrieval

### Hypothesis

Segment-level retrieval followed by local word-window refinement may combine the strong coarse recall of ASR segments with the high localization ceiling of word timestamps.

### Configuration

First stage:

* segment-level TF-IDF retrieval
* top 3 segments retained

Second stage:

* generate local contiguous word windows around those segments
* window sizes: 4, 6, 8, 10, 12, 16, 20 words
* local margin: 10 words

Both stages used the same combined lexical score:

* 0.55 word TF-IDF
* 0.45 character TF-IDF

### Results

Top-1 localization:

* mean tIoU: 0.3117
* median tIoU: 0.2521
* any overlap: 0.6256
* tIoU >= 0.25: 0.5026
* tIoU >= 0.50: 0.3128
* tIoU >= 0.75: 0.1179

Retrieval recall:

* R@1 overlap: 0.6256
* R@3 overlap: 0.7385
* R@5 overlap: 0.7692
* R@10 overlap: 0.7949

Local candidate ceiling:

* oracle mean tIoU: 0.7765
* oracle median tIoU: 0.8974

Top-1 zero-overlap failures:

73 / 195

### Interpretation

The coarse-to-fine decomposition is plausible, but lexical similarity is inadequate for selecting the best local word span.

The large gap between:

* actual mean tIoU: 0.3117
* local candidate oracle mean tIoU: 0.7765

shows that second-stage ranking, rather than timestamp quality, is the dominant failure in this experiment.

The median oracle value of 0.8974 is especially encouraging: for at least half of positive questions the local candidate pool already contains a highly accurate span.

### Decision

DISCARD TF-IDF as the local reranker.

KEEP coarse-to-fine candidate generation as a candidate architecture.

Next experiment: semantic cross-encoder reranking of segment candidates.

---

## EXP-M010 — Cross-Encoder Segment Reranking

### Hypothesis

A semantic cross-encoder reranker may improve evidence-segment ranking beyond lexical TF-IDF, especially where the question and supporting clinical statement are semantically equivalent but lexically different.

### Configuration

First-stage candidate generation:

* segment-level combined TF-IDF
* top 8 lexical segments retained

Semantic reranker:

`cross-encoder/ms-marco-MiniLM-L6-v2`

Device:

CUDA

Maximum sequence length:

256

### Results

Gold-positive questions:

195

Top-1 localization:

* mean tIoU: 0.4347
* median tIoU: 0.4455
* any overlap: 0.7282
* tIoU >= 0.25: 0.6462
* tIoU >= 0.50: 0.4359
* tIoU >= 0.75: 0.2718

Reranked recall:

* R@1 overlap: 0.7282
* R@3 overlap: 0.9231
* R@5 overlap: 0.9436

First-stage top-8 ceiling:

* oracle lexical-top8 mean tIoU: 0.5886

Latency:

* 195-question rerank time: 3.56 s
* mean per question: 0.0182 s

### Comparison

Previous segment TF-IDF:

* mean tIoU: 0.4016
* R@1 overlap: 0.6718
* R@3 overlap: 0.8513

Cross-encoder improvement:

* mean tIoU: +0.0331
* R@1 overlap: +0.0564
* R@3 overlap: +0.0718

### Interpretation

Semantic reranking provides a clear measurable improvement over lexical ranking.

The strongest result is the 0.9231 top-3 overlap recall. This means the correct evidence neighborhood is usually present among the top three semantic segments even when it is not ranked first.

The reranker is also extremely cheap relative to the 60-second request budget.

The remaining localization gap is primarily caused by returning whole ASR segments rather than selecting tight word-level evidence spans.

### Decision

KEEP.

Use semantic segment reranking as the coarse retrieval stage.

Next experiment: semantic word-window refinement inside the top cross-encoder segment neighborhoods.

---

## EXP-M011 — Semantic Word-Window Refinement

### Hypothesis

The MS MARCO cross-encoder that improved segment ranking may also identify the tightest supporting word window inside the top semantic evidence neighborhoods.

### Configuration

Coarse retrieval:

* lexical segment retrieval, top 8
* `cross-encoder/ms-marco-MiniLM-L6-v2` semantic reranking
* top 3 semantic segments retained

Fine candidate generation:

* local contiguous word windows around the top semantic segments
* window sizes from 4 to 24 words

Fine reranking:

`cross-encoder/ms-marco-MiniLM-L6-v2`

### Results

Gold-positive questions:

195

Top-1 localization:

* mean tIoU: 0.3612
* median tIoU: 0.3905
* any overlap: 0.7538
* tIoU >= 0.25: 0.6000
* tIoU >= 0.50: 0.3538
* tIoU >= 0.75: 0.0923

Window recall:

* R@1 overlap: 0.7538
* R@3 overlap: 0.7744
* R@5 overlap: 0.7846

Candidate ceiling:

* oracle local-window mean tIoU: 0.8219
* oracle local-window median tIoU: 0.9059

Latency:

* word-window pairs scored: 73,837
* total rerank time: 30.88 s
* mean per question: 0.1583 s
* estimated mean per 10-question request: 1.5835 s

### Interpretation

The local candidate pool frequently contains an excellent evidence span, but the MS MARCO passage-ranking cross-encoder does not reliably identify the tightest supporting span.

The model remains useful for coarse segment ranking, where EXP-M010 improved retrieval substantially.

Fine word-window semantic reranking with this model reduces mean tIoU relative to simply returning the best semantic segment.

### Decision

DISCARD as the final word-level reranker.

KEEP EXP-M010 semantic segment retrieval.

Pause further evidence-refinement work and establish the first full YES/NO classifier before investing in another localization model.

---

