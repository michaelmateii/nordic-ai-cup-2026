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

## EXP-M012 — Zero-Shot NLI Classification

### Hypothesis

A lightweight NLI model applied to semantically retrieved transcript segments may distinguish positive clinical claims from hard negatives and off-topic questions without task-specific training.

### Configuration

Coarse retrieval:

* lexical segment retrieval, top 8
* `cross-encoder/ms-marco-MiniLM-L6-v2` reranking
* top 5 semantic segments passed to NLI

NLI model:

`cross-encoder/nli-deberta-v3-small`

Input formulation:

* premise: retrieved transcript segment
* hypothesis: raw yes/no question

Decision rule:

Predict YES only if entailment is the highest-probability NLI class for the segment with maximum entailment.

### Initial invalid run

The first evaluation incorrectly converted string answers with:

`bool(row["answer"])`

Because non-empty strings such as `"no"` evaluate to `True`, all 390 gold labels were incorrectly treated as positive.

This run was invalid and was not used for model conclusions.

The evaluation harness was corrected to explicitly parse yes/no strings and now asserts that exactly 195 gold-positive labels are present.

### Corrected results

Overall accuracy:

0.6359

Question-type accuracy:

* positive: 0.2974 (58/195)
* hard_negative: 0.9648 (137/142)
* off_topic: 1.0000 (53/53)

Prediction distribution:

* predicted YES: 0.1615
* gold YES: 0.5000

Confusion matrix:

Rows = gold [NO, YES]
Columns = predicted [NO, YES]

`[[190, 5], [137, 58]]`

Latency:

* 390-question total: 14.72 s
* mean/question: 0.0377 s
* estimated/10-question request: 0.3773 s

### Interpretation

The zero-shot NLI system is strongly conservative.

Negative discrimination is excellent:

* 190/195 negatives correct
* 96.48% hard-negative accuracy
* 100% off-topic accuracy

However, positive recall is only 29.74%.

The current requirement that entailment be the argmax NLI class is therefore probably too strict.

The very low runtime leaves ample room for threshold calibration or improved hypothesis formulation.

### Decision

KEEP as the first classification baseline.

Next experiment: conversation-disjoint calibration of continuous NLI scores before changing the model or hypothesis representation.

---

## EXP-M013 — Conversation-Level NLI Threshold Calibration

### Hypothesis

The poor positive recall in EXP-M012 may primarily be caused by an overly strict NLI decision rule rather than absence of useful entailment signal.

Thresholds must be calibrated conversation-disjoint to avoid question-level leakage between questions from the same consultation.

### Method

Used the cached continuous NLI outputs from EXP-M012.

Five-fold `GroupKFold` validation grouped by `transcript_id`.

For each held-out fold:

* choose the threshold using only the other conversations
* apply that threshold to the held-out conversations
* aggregate out-of-fold predictions across all 390 questions

Tested score formulations:

* maximum entailment
* entailment minus contradiction
* entailment minus maximum other class
* entailment / (entailment + contradiction)

### Results

#### Maximum entailment

* OOF accuracy: 0.7692
* predicted YES rate: 0.5872
* mean threshold: 0.0026
* positive accuracy: 0.8564
* hard_negative accuracy: 0.6338
* off_topic accuracy: 0.8113
* confusion matrix: `[[133, 62], [28, 167]]`

Fold thresholds:

* 0.0021
* 0.0021
* 0.0028
* 0.0029
* 0.0033

#### Entailment minus contradiction

* OOF accuracy: 0.7179
* positive accuracy: 0.4974
* hard_negative accuracy: 0.9155
* off_topic accuracy: 1.0000

#### Entailment minus maximum other class

* OOF accuracy: 0.7282
* positive accuracy: 0.7179
* hard_negative accuracy: 0.7324
* off_topic accuracy: 0.7547

#### Entailment ratio

* OOF accuracy: 0.7333
* positive accuracy: 0.6615
* hard_negative accuracy: 0.7394
* off_topic accuracy: 0.9811

### Interpretation

The NLI model contains substantial useful signal that was hidden by EXP-M012's strict argmax-entailment rule.

Maximum entailment is currently the strongest classification feature.

However, recovering positive recall introduces many false positives:

* 167 / 195 positives recovered
* 62 / 195 negatives incorrectly predicted YES

Hard-negative discrimination is therefore now the dominant classification weakness.

The very low optimum entailment threshold also suggests raw interrogative questions are poorly calibrated as NLI hypotheses.

### Decision

KEEP.

Current classification reference:

`max_entailment` with conversation-disjoint threshold calibration.

Next step: inspect question grammatical forms and convert yes/no questions into declarative hypotheses before rerunning NLI.

---

## EXP-M014 — Declarative-Hypothesis NLI

### Hypothesis

Raw yes/no questions are poorly formed NLI hypotheses. Converting common auxiliary-question structures into declarative clinical claims may improve entailment/contradiction discrimination, particularly for hard negatives.

### Change made

Questions were deterministically converted into declarative hypotheses while preserving their auxiliaries where possible.

Examples of intended transformations:

* `Did the patient report nausea?`
  → `The patient did report nausea.`

* `Does the patient take aspirin?`
  → `The patient does take aspirin.`

* `Was the patient prescribed amoxicillin?`
  → `The patient was prescribed amoxicillin.`

NLI model and retrieval pipeline remained otherwise unchanged.

### Raw uncalibrated results

Overall accuracy:

0.7538

Question-type accuracy:

* positive: 0.5436 (106/195)
* hard_negative: 0.9577 (136/142)
* off_topic: 0.9811 (52/53)

Prediction distribution:

* predicted YES: 0.2897
* gold YES: 0.5000

Confusion matrix:

`[[188, 7], [89, 106]]`

Latency:

* 390 questions: 14.49 s
* mean/question: 0.0372 s
* estimated/10-question request: 0.3715 s

### Conversation-disjoint threshold calibration

Five-fold GroupKFold by conversation.

#### Entailment

* OOF accuracy: 0.8256
* positive: 0.8256
* hard_negative: 0.7887
* off_topic: 0.9245

#### Entailment minus contradiction

* OOF accuracy: 0.7795
* positive: 0.6103
* hard_negative: 0.9296
* off_topic: 1.0000

#### Entailment minus maximum other

* OOF accuracy: 0.8077
* positive: 0.8051
* hard_negative: 0.8310
* off_topic: 0.7547

#### Entailment ratio

Best result.

* OOF accuracy: 0.8513
* predicted YES rate: 0.4641
* positive: 0.8154
* hard_negative: 0.8662
* off_topic: 0.9434
* confusion matrix: `[[173, 22], [36, 159]]`

Fold thresholds:

* 0.1358
* 0.1358
* 0.1311
* 0.1371
* 0.1358

Mean threshold:

0.1351

### Interpretation

Declarative hypotheses materially improve NLI classification.

Compared with EXP-M013's strongest OOF result:

* previous accuracy: 0.7692
* declarative-hypothesis accuracy: 0.8513
* improvement: +0.0821

The best score formulation also becomes substantially more stable across folds.

Hard-negative accuracy of 86.62% confirms that NLI is learning useful contradiction/value mismatch information rather than merely distinguishing topic relevance.

Remaining errors:

* 36 missed positives
* 22 false-positive negatives

### Decision

KEEP.

Current primary classification baseline:

Declarative hypotheses + `entailment_ratio` + conversation-disjoint calibration.

Reference operating threshold:

approximately 0.135.

---

## EXP-M015 — First Composite Development Score

### Goal

Combine the strongest current conversation-disjoint classifier with the strongest current deployable evidence method to estimate an end-to-end development score using the official competition formula.

### Components

Classification:

EXP-M014

* declarative NLI hypotheses
* `entailment_ratio`
* five-fold conversation-disjoint calibration

Evidence:

EXP-M010

* lexical top-8 segment retrieval
* MS MARCO cross-encoder segment reranking
* whole ASR segment returned as evidence

### Results

* Accuracy: 0.8513
* Mean scored tIoU: 0.3719
* Composite score: 0.5636

Gold-positive questions predicted YES:

159 / 195

Among gold-positive questions for which evidence was actually returned:

* mean tIoU: 0.4561

### Interpretation

Classification is substantially stronger than localization.

There are two major sources of evidence-score loss:

1. 36 / 195 gold-positive questions are predicted NO, which forces their tIoU contribution to zero.
2. Whole ASR segments remain too coarse even when the correct evidence neighborhood is found.

The current score should be treated as a development reference rather than a hidden-validation estimate because model and architecture choices have already been informed by the training set.

### Decision

KEEP as the current end-to-end development baseline.

Next step: structured error analysis of the 58 conversation-disjoint classification mistakes before introducing additional rules or models.

---

## EXP-M016 — M014 Classification Error Analysis

### Scope

Analyzed all conversation-disjoint classification errors from EXP-M014 using the best `entailment_ratio` predictions.

### Results

Total errors:

58 / 390

Breakdown:

* false negatives: 36
* false positives: 22

By question type:

* positive false negatives: 36
* hard_negative false positives: 19
* off_topic false positives: 3

Surface features among the 58 errors:

* explicit number in question: 3
* explicit unit in question: 1
* explicit negation term in question: 9

### Important observations

Explicit dose/number/unit mismatches are not the dominant error source.

One off-topic false positive had approximately:

* entailment: 0.0006
* contradiction: 0.0040
* neutral: 0.9979

Despite overwhelmingly neutral evidence, an entailment-to-contradiction ratio can become misleading when both entailment and contradiction probabilities are extremely small.

The declarative converter also produces malformed hypotheses for some passive constructions. Example:

`Were abnormal sounds heard over the lungs?`

was transformed approximately into:

`Abnormal were sounds heard over the lungs.`

### Interpretation

The next highest-value change is not a medication/dose rules engine.

Two higher-priority issues are:

1. decision calibration should reject low-absolute-entailment cases even when the entailment/contradiction ratio is high;
2. passive-question declarative conversion needs improvement.

### Decision

Use a conversation-disjoint two-threshold calibration experiment before changing the underlying NLI model.

---

## EXP-M017 — Entailment-Ratio + Absolute-Entailment Gate

### Hypothesis

Some false positives from EXP-M014 occur when both entailment and contradiction probabilities are extremely small, allowing their ratio to look deceptively positive despite an overwhelmingly neutral NLI output.

Adding a minimum absolute entailment threshold may reject those cases.

### Method

Five-fold conversation-disjoint calibration.

A question is predicted YES only if both:

* entailment ratio >= calibrated ratio threshold
* maximum entailment >= calibrated absolute entailment threshold

Both thresholds are selected using only the training conversations for each fold.

### Results

OOF accuracy:

0.8410

Prediction distribution:

* predicted YES rate: 0.5051

Confusion matrix:

`[[163, 32], [30, 165]]`

Question-type accuracy:

* positive: 0.8462 (165/195)
* hard_negative: 0.8310 (118/142)
* off_topic: 0.8491 (45/53)

Mean calibrated thresholds:

* entailment ratio: 0.046765
* absolute entailment: 0.001228

### Comparison

EXP-M014 ratio-only OOF accuracy:

0.8513

EXP-M017 two-stage OOF accuracy:

0.8410

### Interpretation

Adding an absolute entailment floor does not improve generalization.

Although positive recall increases slightly, both hard-negative and off-topic accuracy fall enough to reduce overall performance.

The fold-specific ratio thresholds also vary considerably, suggesting that this two-dimensional rule is less stable on the small 39-conversation dataset.

A deeper issue remains: EXP-M014 forms the entailment ratio from maximum entailment and maximum contradiction values that may originate from different retrieved transcript segments.

### Decision

DISCARD.

Retain EXP-M014 as the classification reference.

Next experiment: compute NLI evidence/support scores coherently per retrieved segment before aggregating across segments.

---

## EXP-M018 — Coherent Segmentwise NLI Aggregation

### Hypothesis

Previous NLI aggregation combined maximum entailment and maximum contradiction values that could originate from different retrieved transcript segments.

Computing support scores coherently within each individual segment may improve classification.

### Configuration

For each of the top five retrieved transcript segments, compute:

* entailment probability
* contradiction probability
* neutral probability

Then derive segment-specific scores.

Tested:

* `max_segment_ratio`
* `max_segment_entailment`
* `max_segment_margin`
* `max_segment_vs_other`

All thresholds were calibrated with five-fold conversation-disjoint GroupKFold validation.

### Raw segment-ratio result

Using a fixed `max_segment_ratio >= 0.5` rule:

* accuracy: 0.8590
* positive: 0.8974
* hard_negative: 0.7746
* off_topic: 0.9434

### Conversation-disjoint calibration

#### max_segment_ratio

* OOF accuracy: 0.8615
* positive: 0.8923
* hard_negative: 0.7817
* off_topic: 0.9623
* predicted YES rate: 0.5308

#### max_segment_entailment

* OOF accuracy: 0.8256
* positive: 0.8256
* hard_negative: 0.7887
* off_topic: 0.9245

#### max_segment_margin

BEST RESULT.

Score definition:

`entailment - contradiction`

computed within the same retrieved transcript segment.

Results:

* OOF accuracy: 0.8821
* predicted YES rate: 0.4897
* positive: 0.8718 (170/195)
* hard_negative: 0.8662 (123/142)
* off_topic: 0.9623 (51/53)
* confusion matrix: `[[174, 21], [25, 170]]`

Fold thresholds:

* 0.000585
* 0.000585
* 0.000585
* 0.000441
* 0.000585

Mean threshold:

0.000556

#### max_segment_vs_other

* OOF accuracy: 0.7744
* positive: 0.6000
* hard_negative: 0.9437
* off_topic: 0.9623

### Comparison with previous best

EXP-M014:

* OOF accuracy: 0.8513

EXP-M018:

* OOF accuracy: 0.8821

Improvement:

+0.0308 absolute accuracy

### Interpretation

Coherent per-segment NLI aggregation materially improves classification.

The best feature is the maximum segment-level entailment-minus-contradiction margin.

This improves positive recall while preserving strong hard-negative and off-topic discrimination.

The result also confirms that independently aggregating entailment and contradiction across unrelated transcript segments was suboptimal.

### Decision

KEEP.

Current primary classification method:

`max_segment_margin`

with conversation-disjoint calibrated threshold approximately:

`0.00056`

---

## EXP-M019 — M018 Composite Development Score

### Goal

Measure the end-to-end development effect of replacing EXP-M014 classification with the stronger coherent segmentwise NLI classifier from EXP-M018 while keeping the evidence-localization method fixed.

### Components

Classification:

* EXP-M018
* `max_segment_margin`
* five-fold conversation-disjoint calibration

Evidence:

* EXP-M010
* cross-encoder segment retrieval
* whole ASR segment returned as evidence

### Results

* Accuracy: 0.8821
* Mean scored tIoU: 0.3816
* Composite score: 0.5818

Gold-positive questions predicted YES:

170 / 195

Among returned gold-positive spans:

* mean tIoU: 0.4378

### Comparison

EXP-M015:

* Accuracy: 0.8513
* Mean scored tIoU: 0.3719
* Composite score: 0.5636

EXP-M019:

* Accuracy: 0.8821
* Mean scored tIoU: 0.3816
* Composite score: 0.5818

Composite improvement:

+0.0182

### Interpretation

The stronger classifier improves both accuracy and overall scored tIoU by recovering additional gold-positive questions.

However, localization remains the dominant bottleneck.

Classification is now 88.21% accurate, while evidence spans returned for correctly detected positives average only 0.4378 tIoU despite an ASR word-boundary oracle ceiling of 0.9294.

### Decision

KEEP as the current end-to-end development reference.

Shift primary optimization effort to evidence-span localization.

---

## EXP-M020 — Extractive-QA Evidence Anchor

### Hypothesis

A lightweight SQuAD-style extractive QA model may identify a tight answer phrase inside the top semantic evidence neighborhoods, which can then be mapped back to faster-whisper word timestamps.

### Model

`deepset/minilm-uncased-squad2`

### Results

Mean tIoU by answer-span padding:

* ±0 words: 0.2021
* ±1 word: 0.1951
* ±2 words: 0.2094
* ±4 words: 0.2134
* ±6 words: 0.2007

Best configuration:

±4 words

* mean tIoU: 0.2134
* median tIoU: 0.1391
* any overlap: 0.5487
* tIoU >= 0.50: 0.1487

Latency:

* ~0.094 s/question
* ~0.944 s/10 questions

### Comparison

M010 whole semantic segment:

0.4347 mean tIoU

M020 best extractive QA:

0.2134 mean tIoU

### Interpretation

SQuAD-style extractive QA is poorly matched to the evidence-localization objective.

The model tends to extract a short answer entity or phrase, while the annotation represents the fuller supporting clinical statement.

Padding the answer does not recover enough of the gold interval.

### Decision

DISCARD.

---

## EXP-M021 — NLI-Margin Segment Evidence

### Hypothesis

The transcript segment producing the strongest segmentwise NLI entailment-minus-contradiction margin may also be a better evidence span than the generic MS MARCO passage-ranking winner.

### Results

Positive questions:

195

Localization:

* mean tIoU: 0.3965
* median tIoU: 0.4029
* any overlap: 0.6564
* tIoU >= 0.25: 0.5692
* tIoU >= 0.50: 0.4051
* tIoU >= 0.75: 0.2462

### Comparison

M010 cross-encoder segment:

0.4347 mean tIoU

M021 NLI-margin segment:

0.3965 mean tIoU

### Interpretation

The segment that best supports YES/NO classification is not necessarily the segment whose boundaries best match the annotated evidence interval.

NLI segment scoring remains valuable for classification but should not replace the M010 passage-ranking evidence selector.

### Decision

DISCARD as evidence-selection method.

KEEP M018 NLI-margin scoring for classification.

KEEP M010 MS MARCO segment retrieval as the evidence baseline.

---

## EXP-M022 — NLI Word-Span Refinement

### Hypothesis

The NLI model that performs strongly for YES/NO classification may rank local word windows better than the MS MARCO passage-ranking model.

### Configuration

* top 8 lexical ASR segments
* MS MARCO reranking
* top 3 semantic segment neighborhoods
* local word windows from 4 to 28 words
* declarative question-to-claim conversion
* NLI entailment-minus-contradiction margin used to rank windows

### Results

* mean tIoU: 0.3153
* median tIoU: 0.2779
* any overlap: 0.7077
* tIoU >= 0.25: 0.5641
* tIoU >= 0.50: 0.2821
* tIoU >= 0.75: 0.1077

Retrieval recall:

* R@1: 0.7077
* R@3: 0.7692
* R@5: 0.7795

Candidate ceiling:

* oracle mean tIoU: 0.8219
* oracle median tIoU: 0.9059

Latency:

* 75,201 NLI window pairs
* 188.54 seconds total
* 0.9669 seconds/question
* 9.6687 seconds/10-question request

### Comparison

* M010 whole semantic segment: 0.4347
* M011 MS MARCO word refinement: 0.3612
* M022 NLI word refinement: 0.3153

### Interpretation

NLI is effective for claim classification but poorly suited to selecting exact evidence boundaries from a large set of overlapping word windows.

Arbitrary fixed-size sliding windows remain a poor representation of the annotated evidence despite their high oracle ceiling.

The computational cost is also unnecessarily high.

### Decision

DISCARD.

Stop pursuing brute-force sliding-word-window reranking.

---

## EXP-M023 — Sentence/Utterance Candidate Ceiling

### Hypothesis

Gold evidence intervals may correspond more closely to complete punctuation-delimited spoken statements than to fixed-size word windows.

### Candidate construction

Candidates were generated from faster-whisper word timestamps using punctuation boundaries.

Tested contiguous groups of:

* 1 sentence
* up to 2 sentences
* up to 3 sentences

Gold evidence timestamps were used only to calculate the oracle ceiling.

### Results

#### Up to 1 sentence

* mean oracle tIoU: 0.7100
* median oracle tIoU: 0.8000
* tIoU >= 0.50: 0.7641
* tIoU >= 0.75: 0.5436
* tIoU >= 0.90: 0.3487

#### Up to 2 sentences

* mean oracle tIoU: 0.7854
* median oracle tIoU: 0.8919
* tIoU >= 0.50: 0.8513
* tIoU >= 0.75: 0.6974
* tIoU >= 0.90: 0.4718

#### Up to 3 sentences

* mean oracle tIoU: 0.7997
* median oracle tIoU: 0.8974
* tIoU >= 0.50: 0.8718
* tIoU >= 0.75: 0.7179
* tIoU >= 0.90: 0.4974

### Interpretation

Sentence/utterance candidates capture much more of the annotation geometry than whole ASR segments while avoiding the huge redundant candidate space of sliding word windows.

Three-sentence groups provide only a modest oracle improvement over two-sentence groups, but the combined 1–3 sentence candidate set has a useful ceiling near 0.80 mean tIoU.

### Decision

KEEP candidate representation.

Next experiment: directly rank all 1–3 sentence candidates with the MS MARCO cross-encoder.

---

## EXP-M024 — Sentence-Group Cross-Encoder Retrieval

### Hypothesis

The sentence/utterance candidate representation from EXP-M023 has a high oracle localization ceiling. Ranking all 1–3 sentence groups with the MS MARCO cross-encoder may exploit that improved candidate geometry.

### Results

Top-1 localization:

* mean tIoU: 0.3576
* median tIoU: 0.3095
* any overlap: 0.7026
* tIoU >= 0.25: 0.5949
* tIoU >= 0.50: 0.2974
* tIoU >= 0.75: 0.1538

Retrieval recall:

* R@1: 0.7026
* R@3: 0.7846
* R@5: 0.8308

Candidate ceiling:

* oracle mean tIoU: 0.7997
* oracle median tIoU: 0.8974

Latency:

* 30,849 pairs scored
* total rerank time: 16.21 s
* mean/question: 0.0831 s
* estimated/10-question request: 0.8313 s

### Comparison

* M010 segment localization: 0.4347
* M024 sentence-group localization: 0.3576

### Interpretation

Sentence-group candidates remain promising, but generic passage relevance does not reliably select the annotation-compatible boundaries.

The large gap between actual and oracle tIoU indicates a ranking problem rather than a candidate-generation problem.

### Decision

DISCARD MS MARCO as the final sentence-group ranker.

KEEP the 1–3 sentence candidate representation.

Next experiment: supervised conversation-disjoint ranking of sentence candidates using only inference-available features.

---

## EXP-M025 — Supervised Sentence Span Ranker

### Hypothesis

The high oracle quality of 1–3 sentence evidence candidates may be exploitable with a lightweight supervised ranker trained on inference-available features rather than another generic zero-shot reranker.

### Method

Candidates:

* contiguous 1–3 sentence groups derived from faster-whisper punctuation boundaries

Features available at inference time:

* sentence count
* word count
* candidate duration
* lexical TF-IDF similarity
* MS MARCO semantic relevance
* temporal IoU with M010 coarse anchor
* distance from M010 anchor midpoint
* start-distance from anchor
* end-distance from anchor
* whether candidate contains anchor midpoint

Training target:

* temporal IoU against annotated gold evidence

Model:

`HistGradientBoostingRegressor`

Validation:

* 5-fold GroupKFold
* grouped by `transcript_id`
* no conversation appears in both training and held-out ranking evaluation

### Results

Questions:

195 positive questions

OOF localization:

* mean tIoU: 0.4771
* median tIoU: 0.4766
* any overlap: 0.7231
* tIoU >= 0.25: 0.6872
* tIoU >= 0.50: 0.4769
* tIoU >= 0.75: 0.3333

Candidate ceiling:

* oracle mean tIoU: 0.7997
* oracle median tIoU: 0.8974

Experiment wall time:

25.26 seconds

### Comparison

* M010 zero-shot segment: 0.4347
* M024 zero-shot sentence groups: 0.3576
* M025 supervised sentence groups: 0.4771

Improvement over previous best localization:

+0.0424 mean tIoU

### Interpretation

Task-specific supervision materially improves evidence ranking.

The result validates both:

1. sentence/utterance candidate geometry;
2. conversation-disjoint supervised ranking.

There remains a large ranking gap between the 0.4771 achieved tIoU and 0.7997 sentence-candidate oracle, indicating further feature and ranking improvements may be valuable.

Because model selection has already been influenced by the supplied training set, this OOF result should be treated as a development estimate rather than an unbiased hidden-validation prediction.

### Decision

KEEP.

Current primary evidence-localization method:

supervised 1–3 sentence candidate ranker.

Next step: combine M025 evidence with M018 classification and recompute the development composite score.

---

## EXP-M026 — M025 Composite Development Score

### Goal

Measure the combined development score of the strongest current classifier and strongest conversation-disjoint evidence-localization method.

### Components

Classification:

EXP-M018

* declarative NLI hypotheses
* coherent segmentwise NLI
* `max_segment_margin = entailment - contradiction`
* conversation-disjoint calibrated operating threshold

Evidence localization:

EXP-M025

* 1–3 sentence evidence candidates
* supervised HistGradientBoosting ranker
* conversation-disjoint GroupKFold evaluation
* inference-available features only

### Results

* Accuracy: 0.8821
* Mean scored tIoU: 0.4183
* Composite score: 0.6038

Gold-positive questions predicted YES:

170 / 195

Among returned gold-positive evidence spans:

* mean tIoU: 0.4798

### Comparison

EXP-M019:

* Accuracy: 0.8821
* Mean scored tIoU: 0.3816
* Composite: 0.5818

EXP-M026:

* Accuracy: 0.8821
* Mean scored tIoU: 0.4183
* Composite: 0.6038

Improvement:

+0.0220 composite score

### Interpretation

The supervised sentence evidence ranker provides a measurable end-to-end gain without changing classification.

Classification and evidence localization have now both been validated conversation-disjoint on the supplied training conversations.

However, repeated architecture decisions have used this same development corpus, so further local optimization risks overfitting model-selection decisions.

### Decision

KEEP.

Current best development pipeline.

Next priority: create a deployable full-data version of the supervised evidence ranker and run the official 19-conversation validation before further local optimization.

---

## EXP-M027 — Full-Data Deployment Evidence Ranker

### Goal

Convert the conversation-disjoint evidence-localization experiment from EXP-M025 into a single deployment model that can be used on the official hidden validation and evaluation conversations.

### Background

EXP-M025 evaluated a supervised sentence-span ranker with five-fold GroupKFold validation grouped by `transcript_id`.

Its out-of-fold localization result was:

* mean tIoU: 0.4771
* median tIoU: 0.4766
* any overlap: 0.7231
* tIoU >= 0.25: 0.6872
* tIoU >= 0.50: 0.4769
* tIoU >= 0.75: 0.3333

Sentence-candidate oracle:

* mean tIoU: 0.7997
* median tIoU: 0.8974

### Change made

After preserving the EXP-M025 out-of-fold evaluation, trained a final `HistGradientBoostingRegressor` on the complete set of supplied positive training questions.

Training set:

* 195 positive questions
* all associated 1–3 sentence evidence candidates

Features:

* `sentence_count`
* `word_count`
* `duration`
* `lexical_score`
* `semantic_score`
* `anchor_iou`
* `anchor_mid_distance`
* `anchor_start_distance`
* `anchor_end_distance`
* `contains_anchor_mid`

Model configuration:

* learning rate: 0.06
* max iterations: 250
* max leaf nodes: 15
* min samples leaf: 30
* L2 regularization: 1.0
* random state: 42

Each training question contributes equal total sample weight regardless of its number of candidate spans.

### Deployment artifact

Saved locally as:

`medical/artifacts/models/sentence_ranker.joblib`

Artifact verification:

* bundle loads successfully with `joblib`
* `max_sentences`: 3
* `training_questions`: 195
* expected 10-feature schema present

The deployment artifact is intentionally excluded from Git under `medical/artifacts/`.

### Validation of training change

Rerunning the script after adding full-data model training preserved the original EXP-M025 out-of-fold score:

* mean tIoU: 0.4771
* median tIoU: 0.4766
* candidate oracle mean: 0.7997

Therefore the deployment-training addition did not alter the measured OOF experiment.

### Interpretation

EXP-M027 does not claim a new development-score improvement.

Its purpose is operational: produce one ranker trained on all available labeled training data for use against previously unseen official validation/evaluation conversations.

The unbiased development reference remains EXP-M025's conversation-disjoint OOF result.

### Decision

KEEP.

Deployment evidence ranker is ready for integration into the competition runtime.

---

## EXP-M028B — End-to-End HTTP Smoke Test

### Goal

Verify the frozen Medical pipeline through the actual FastAPI `/predict` interface before using an official validation attempt.

### Test case

Training consultation:

`conversation_sample_10.mp3`

Questions:

10

Full runtime path:

audio Base64
→ faster-whisper ASR
→ lexical + MS MARCO retrieval
→ segmentwise DeBERTa NLI
→ supervised M025 sentence evidence ranker
→ competition JSON response

### Operational results

* HTTP status: 200
* request latency: 6.71 seconds
* response structure: PASS
* answer count: 10
* evidence_start count: 10
* evidence_end count: 10
* FALSE answers correctly returned null evidence
* TRUE answers returned finite ordered evidence intervals

### Prediction result

* accuracy: 9/10 = 0.900
* predicted YES: 4/10

Single classification error:

`Were abnormal sounds heard over the lungs?`

* gold: FALSE
* predicted: TRUE
* returned evidence: 53.54–55.88 s

The following positive question correctly selected the same evidence region:

`Were the lungs and heart normal on auscultation?`

* gold: TRUE
* predicted: TRUE
* returned evidence: 53.54–55.88 s

### Interpretation

The HTTP/runtime infrastructure is functioning correctly and comfortably within the 60-second request timeout.

The remaining error exposes a previously identified claim-conversion weakness for passive `was/were` questions rather than an evidence-retrieval failure.

### Decision

KEEP runtime/infrastructure.

Do not use an official validation attempt until passive claim construction is tested and either accepted or rejected using conversation-disjoint OOF evaluation.

---

## EXP-M028C — Passive `is/are/was/were` Claim Conversion

### Motivation

The end-to-end smoke test exposed a known declarative-conversion failure:

`Were abnormal sounds heard over the lungs?`

The existing converter could produce a malformed hypothesis similar to:

`Abnormal were sounds heard over the lungs.`

A targeted passive/copular conversion rule was tested.

### Change

Added predicate-start detection for passive/copular questions such as:

* `Were abnormal sounds heard over the lungs?`
* `Were bacteria identified in the sample?`
* `Was the patient prescribed amoxicillin?`
* `Is the stomach acid medication being discontinued?`

The intention was to generate grammatical hypotheses such as:

`Abnormal sounds were heard over the lungs.`

### Validation

Five-fold conversation-disjoint GroupKFold calibration was rerun on all 390 questions.

Best classification method remained:

`max_segment_margin`

### Results

EXP-M028C:

* OOF accuracy: 0.8795
* predicted YES rate: 0.4872
* positive accuracy: 0.8667 (169/195)
* hard_negative accuracy: 0.8662 (123/142)
* off_topic accuracy: 0.9623 (51/53)
* confusion matrix: `[[174, 21], [26, 169]]`

Previous EXP-M018 reference:

* OOF accuracy: 0.8821
* positive accuracy: 0.8718 (170/195)
* hard_negative accuracy: 0.8662
* off_topic accuracy: 0.9623

Difference:

* overall accuracy: -0.0026
* one additional false-negative positive question
* no improvement in hard-negative or off-topic performance

### Interpretation

The targeted linguistic correction improves grammaticality but does not improve conversation-disjoint classification performance.

The visible smoke-test error should not be patched at the cost of measured generalization.

### Decision

DISCARD.

Revert the passive conversion change and retain the EXP-M018 claim converter and operating point for official validation.

---

## EXP-M028D — Full 39-Conversation Local End-to-End Evaluation

### Goal

Validate the exact frozen deployment runtime against all supplied Medical Appointment conversations through the official `local_evaluator.py`.

This test exercises:

audio Base64 request
→ FastAPI `/predict`
→ faster-whisper ASR
→ lexical + MS MARCO retrieval
→ segmentwise DeBERTa NLI classification
→ supervised sentence evidence localization
→ response validation
→ official scoring

### Runtime

Machine:

Windows PC — NVIDIA GeForce GTX 1060 6 GB

Endpoint:

`http://127.0.0.1:8000/predict`

ASR:

* faster-whisper
* `distil-large-v3`
* CUDA
* `int8_float32`
* beam size 1
* word timestamps enabled

Classification:

* `cross-encoder/nli-deberta-v3-small`
* declarative claim transformation
* maximum segment-level entailment-minus-contradiction margin
* frozen threshold: 0.000585

Evidence:

* 1–3 punctuation-delimited sentence candidates
* full-data EXP-M027 HistGradientBoosting sentence ranker

### Attempt statistics

* questions: 390
* correct: 340
* unanswered: 0
* conversations: 39
* failed conversations: 0
* timeouts: 0

### Accuracy by question type

* positive: 0.851 (166/195)
* hard_negative: 0.866 (123/142)
* off_topic: 0.962 (51/53)

### Evidence localization

* mean scored tIoU: 0.419
* no span returned for gold positives: 29
* tIoU among answered-YES gold positives: 0.492
* diagnostic answered-YES positive count: 166

### Runtime

* mean round trip per conversation: 6510 ms
* worst round trip: 11537 ms
* mean per question: 651 ms
* worst request used approximately 19% of the 60-second budget

### Final score

* Accuracy: 0.872
* Mean tIoU: 0.419
* Composite score: 0.600

### Comparison

EXP-M026 OOF-derived development reference:

* Accuracy: 0.8821
* Mean scored tIoU: 0.4183
* Composite: 0.6038

Exact deployed local runtime:

* Accuracy: 0.872
* Mean scored tIoU: 0.419
* Composite: 0.600

### Interpretation

The exact deployed runtime closely matches the conversation-disjoint development estimate.

There were no failed conversations, malformed responses, or timeouts.

Evidence localization reproduces the development estimate almost exactly.

Classification is approximately one percentage point lower than the OOF reference but remains close enough that additional tuning on the supplied 39 conversations would carry greater overfitting risk than likely benefit.

Worst-case latency of 11.54 seconds is comfortably below the 60-second per-request timeout.

### Decision

KEEP and FREEZE for first official validation.

Do not make further model-selection changes before obtaining an external validation score.

---

## EXP-M029 — First Official Hidden Validation

### Goal

Measure the frozen Medical Appointment pipeline on the official hidden validation set before using the single allowed evaluation attempt.

### Deployment

Public endpoint:

ngrok HTTPS tunnel → Windows GTX 1060 Medical FastAPI endpoint

Pipeline:

* faster-whisper `distil-large-v3`
* CUDA `int8_float32`
* beam size 1
* word timestamps
* lexical TF-IDF retrieval
* MS MARCO MiniLM reranking
* DeBERTa-v3-small segmentwise NLI classification
* frozen max-segment-margin operating threshold
* supervised 1–3 sentence evidence ranker trained on all supplied positive training examples

### Official result

* Score: **0.5843964099377816**
* Service errors: **0**
* Attempt completed successfully

Submitted:

2026-09-17 23:11:13 UTC

Finished:

2026-09-17 23:13:34 UTC

### Comparison

OOF development composite:

* 0.6038

Full supplied-data endpoint score:

* 0.6000

Official hidden validation:

* 0.5844

Difference versus local deployed score:

* -0.0156 absolute

### Interpretation

The hidden-validation score is close to both the conversation-disjoint development estimate and the full local endpoint score.

There is no evidence of catastrophic train/validation overfitting.

The endpoint completed successfully without service errors, confirming that the deployment architecture, public tunnel, request handling, response schema, and runtime are operational under the official evaluator.

The small hidden-set degradation is consistent with normal generalization loss.

The official validation score should now be treated as the strongest available external estimate of final performance.

### Decision

KEEP as the current frozen competition system.

Do not spend the single evaluation attempt yet.

Only replace this system if a subsequent change shows a convincing conversation-disjoint improvement locally and has a clear technical justification.

---

## EXP-M030 — Fine-Tuned Sentence Cross-Encoder

### Hypothesis

The generic MS MARCO cross-encoder failed to exploit the high-quality sentence candidate space zero-shot, but task-specific fine-tuning on temporal-IoU supervision may learn the Medical Appointment evidence-ranking objective directly.

### Model

Base:

`cross-encoder/ms-marco-MiniLM-L6-v2`

Training target:

candidate temporal IoU with annotated evidence

Candidate space:

contiguous 1–3 sentence groups derived from faster-whisper word timestamps

Training candidate sampling:

* positive / near-positive candidates
* high-scoring hard negatives
* random negatives

### Validation methodology

5-fold GroupKFold grouped by `transcript_id`.

No consultation appears in both training and held-out evaluation within a fold.

Held-out gold timestamps are used only for scoring.

### Results

195 gold-positive questions.

OOF localization:

* mean tIoU: 0.5089
* median tIoU: 0.5274
* any overlap: 0.7590
* tIoU >= 0.25: 0.7128
* tIoU >= 0.50: 0.5128
* tIoU >= 0.75: 0.3692

Candidate oracle:

* mean tIoU: 0.7997

Wall time:

* 368.0 seconds

### Comparison

EXP-M025 supervised HGB:

* mean tIoU: 0.4771

EXP-M030 fine-tuned cross-encoder:

* mean tIoU: 0.5089

Improvement:

* +0.0318 absolute mean tIoU
* approximately +6.7% relative

### Interpretation

Task-specific neural supervision meaningfully improves evidence ranking over both the generic zero-shot cross-encoder and the handcrafted-feature gradient boosting model.

The improvement appears across mean, median, overlap rate, and high-IoU thresholds.

A substantial gap remains to the 0.7997 candidate oracle, leaving room for ranking improvements and model ensembling.

### Decision

KEEP.

Next step: combine EXP-M030 evidence with the frozen EXP-M018 classifier and calculate conversation-disjoint composite score before producing a deployment model.

---

## EXP-M031 — M030 Composite Development Score

### Goal

Measure the end-to-end development score from combining the frozen EXP-M018 classifier with the improved EXP-M030 fine-tuned neural evidence ranker.

### Components

Classification:

EXP-M018 `max_segment_margin`

* OOF accuracy: 0.8821
* positive accuracy: 0.8718
* hard-negative accuracy: 0.8662
* off-topic accuracy: 0.9623

Evidence:

EXP-M030 fine-tuned sentence cross-encoder

* OOF mean evidence tIoU: 0.5089

### Results

* Accuracy: 0.8821
* Mean scored tIoU: 0.4428
* Composite score: 0.6185

Gold-positive questions predicted YES:

170 / 195

Mean tIoU among returned gold-positive spans:

0.5079

### Comparison

EXP-M026:

* Accuracy: 0.8821
* Mean scored tIoU: 0.4183
* Composite: 0.6038

EXP-M031:

* Accuracy: 0.8821
* Mean scored tIoU: 0.4428
* Composite: 0.6185

Composite improvement:

+0.0147

### Interpretation

Task-specific neural evidence ranking produces a meaningful end-to-end improvement while holding classification constant.

The result is conversation-disjoint and therefore materially stronger evidence than an improvement measured only on the full supplied training set.

### Decision

KEEP.

Current best OOF development system.

Next: measure complementarity between the M025 HGB ranker and M030 neural ranker before investing in a candidate-level ensemble.
## EXP-M031 — M030 Composite Development Score

### Goal

Measure the end-to-end development score from combining the frozen EXP-M018 classifier with the improved EXP-M030 fine-tuned neural evidence ranker.

### Components

Classification:

EXP-M018 `max_segment_margin`

* OOF accuracy: 0.8821
* positive accuracy: 0.8718
* hard-negative accuracy: 0.8662
* off-topic accuracy: 0.9623

Evidence:

EXP-M030 fine-tuned sentence cross-encoder

* OOF mean evidence tIoU: 0.5089

### Results

* Accuracy: 0.8821
* Mean scored tIoU: 0.4428
* Composite score: 0.6185

Gold-positive questions predicted YES:

170 / 195

Mean tIoU among returned gold-positive spans:

0.5079

### Comparison

EXP-M026:

* Accuracy: 0.8821
* Mean scored tIoU: 0.4183
* Composite: 0.6038

EXP-M031:

* Accuracy: 0.8821
* Mean scored tIoU: 0.4428
* Composite: 0.6185

Composite improvement:

+0.0147

### Interpretation

Task-specific neural evidence ranking produces a meaningful end-to-end improvement while holding classification constant.

The result is conversation-disjoint and therefore materially stronger evidence than an improvement measured only on the full supplied training set.

### Decision

KEEP.

Current best OOF development system.

Next: measure complementarity between the M025 HGB ranker and M030 neural ranker before investing in a candidate-level ensemble.
## EXP-M031 — M030 Composite Development Score

### Goal

Measure the end-to-end development score from combining the frozen EXP-M018 classifier with the improved EXP-M030 fine-tuned neural evidence ranker.

### Components

Classification:

EXP-M018 `max_segment_margin`

* OOF accuracy: 0.8821
* positive accuracy: 0.8718
* hard-negative accuracy: 0.8662
* off-topic accuracy: 0.9623

Evidence:

EXP-M030 fine-tuned sentence cross-encoder

* OOF mean evidence tIoU: 0.5089

### Results

* Accuracy: 0.8821
* Mean scored tIoU: 0.4428
* Composite score: 0.6185

Gold-positive questions predicted YES:

170 / 195

Mean tIoU among returned gold-positive spans:

0.5079

### Comparison

EXP-M026:

* Accuracy: 0.8821
* Mean scored tIoU: 0.4183
* Composite: 0.6038

EXP-M031:

* Accuracy: 0.8821
* Mean scored tIoU: 0.4428
* Composite: 0.6185

Composite improvement:

+0.0147

### Interpretation

Task-specific neural evidence ranking produces a meaningful end-to-end improvement while holding classification constant.

The result is conversation-disjoint and therefore materially stronger evidence than an improvement measured only on the full supplied training set.

### Decision

KEEP.

Current best OOF development system.

Next: measure complementarity between the M025 HGB ranker and M030 neural ranker before investing in a candidate-level ensemble.

---

## EXP-M032A — Evidence Ranker Complementarity

### Goal

Determine whether the EXP-M025 HGB evidence ranker and EXP-M030 fine-tuned neural ranker make sufficiently different localization errors to justify an ensemble.

### Results

Individual systems:

* M025 HGB mean tIoU: 0.4771
* M030 neural mean tIoU: 0.5089

Selected-span agreement:

* same span: 87/195 = 0.4462

Comparison:

* neural better: 50 questions
* HGB better: 38 questions
* tie: 107 questions

Oracle selector between the two selected spans:

* mean tIoU: 0.5862
* median tIoU: 0.6316
* tIoU >= 0.50: 0.6205
* tIoU >= 0.75: 0.4513

Oracle gain over M030:

+0.0773 mean tIoU

### Interpretation

The two rankers are substantially complementary.

Although M030 is stronger on average, M025 selects a better span for 38/195 questions.

An ideal selector between only these two outputs would raise evidence localization from 0.5089 to 0.5862 without changing candidate generation.

This is enough potential gain to justify a confidence-based selector before attempting additional expensive model training.

### Decision

KEEP.

Proceed to a conversation-disjoint selector using only inference-available confidence signals.

---

## EXP-M032B — Confidence-Gated Evidence Selector

### Hypothesis

The complementary M025 and M030 evidence rankers may be combined by choosing between their top spans according to their confidence difference.

### Validation

A single confidence-difference threshold was calibrated conversation-disjoint using 5-fold GroupKFold.

### Results

Evidence OOF:

* M025 HGB: 0.4771
* M030 neural: 0.5089
* raw confidence selector: 0.5195
* CV-calibrated selector: 0.5052
* oracle of the two selected spans: 0.5862

CV selector:

* neural selected: 121/195
* HGB selected: 74/195

Fold thresholds:

* -0.208334
* +0.084716
* +0.023157
* +0.023157
* +0.023157

Composite:

* classification accuracy: 0.8821
* mean scored tIoU: 0.4375
* composite: 0.6153

Reference M031 composite:

0.6185

### Interpretation

The rankers are complementary, but their raw confidence scales are not consistently comparable across folds.

Threshold calibration is unstable and reduces OOF performance below M030 alone.

The raw 0.5195 result is diagnostic but is not sufficient justification for deployment

---

## EXP-M033 — Candidate-Level Score Fusion

### Hypothesis

Rank-normalized candidate-level fusion of the M025 HGB ranker and M030 fine-tuned neural ranker may exploit their complementary errors better than choosing only between their final top spans.

### Validation

5-fold conversation-disjoint calibration of the neural fusion weight.

### Results

OOF evidence:

* mean tIoU: 0.5035
* median tIoU: 0.5367
* any overlap: 0.7128
* tIoU >= 0.50: 0.5128
* tIoU >= 0.75: 0.4000

Fold neural weights:

* 0.85
* 0.50
* 0.95
* 0.50
* 0.50

Mean neural weight:

0.660

Composite:

* classification accuracy: 0.8821
* mean scored tIoU: 0.4337
* composite score: 0.6130

References:

* M030 evidence: 0.5089
* M031 composite: 0.6185
* candidate oracle: 0.7997

### Interpretation

Candidate-level fusion does not improve mean localization relative to M030.

The preferred fusion weight varies substantially across folds, indicating limited stability.

Although the proportion of high-quality spans at tIoU >= 0.75 improves, the mean score decreases, which is what matters for the official metric.

### Decision

DISCARD.

Keep M030 as the evidence-ranking reference.

---

## EXP-M034 — Hybrid Candidate Geometry Ceiling

### Goal

Measure whether supplementing sentence candidates with short word-window candidates increases the evidence-localization ceiling.

### Candidate space

Existing candidates:

* 1 sentence
* 2 consecutive sentences
* 3 consecutive sentences

Additional word windows:

* 6 words
* 10 words
* 14 words
* 18 words
* 24 words
* 32 words

Word-window stride:

3 words

### Results

Oracle localization:

* mean tIoU: 0.8541
* median tIoU: 0.9127
* minimum tIoU: 0.2353
* tIoU >= 0.50: 0.9590
* tIoU >= 0.75: 0.8256
* tIoU >= 0.90: 0.5590

Candidate volume:

* mean candidates per conversation: 760.5
* maximum candidates per conversation: 1468

Oracle winner types:

* sentence_1: 107
* sentence_2: 28
* sentence_3: 4
* word_6: 28
* word_10: 18
* word_14: 4
* word_18: 3
* word_24: 2
* word_32: 1

### Comparison

Sentence-only oracle:

0.7997

Hybrid oracle:

0.8541

Improvement:

+0.0544 mean tIoU

Unrestricted word-timestamp oracle:

0.9294

### Interpretation

Short word windows materially improve candidate geometry.

Most of the additional oracle wins come from 6-word and 10-word windows. Larger word-window sizes contribute comparatively few unique wins while substantially increasing candidate volume.

The full hybrid candidate space is too large to score exhaustively with a neural cross-encoder within the competition latency budget.

### Decision

KEEP candidate-geometry direction.

Next: determine the smallest hybrid candidate configuration that preserves most of the 0.8541 oracle ceiling.

---

## EXP-M035 — Hybrid Candidate Ablation

### Goal

Find the smallest evidence candidate space that preserves most of the localization ceiling gained in EXP-M034.

### Results

Sentence-only:

* oracle mean tIoU: 0.7997
* mean candidates: 158.2
* max candidates: 249

Sentences + 6-word windows:

* oracle mean tIoU: 0.8332
* mean candidates: 261.0
* max candidates: 457

Sentences + 6- and 10-word windows:

* oracle mean tIoU: 0.8447
* mean candidates: 362.7
* max candidates: 663

Sentences + 6-, 10-, and 14-word windows:

* oracle mean tIoU: 0.8481
* mean candidates: 463.8
* max candidates: 867

Sentences + 6-, 10-, 14-, and 18-word windows:

* oracle mean tIoU: 0.8516
* mean candidates: 564.3
* max candidates: 1067

Full hybrid:

* oracle mean tIoU: 0.8541
* mean candidates: 760.5
* max candidates: 1468

### Interpretation

The majority of the candidate-geometry improvement comes from short 6- and 10-word windows.

`sentences + w6 + w10` improves the oracle from 0.7997 to 0.8447, recovering approximately 83% of the total EXP-M034 oracle improvement while using less than half the candidate volume of the full hybrid set.

Larger word windows provide diminishing returns relative to their inference cost.

### Decision

KEEP `sentence + word_6 + word_10` as the next candidate space.

Do not add larger word-window families unless the compact hybrid model demonstrates that candidate geometry is still the dominant bottleneck.


---

## EXP-M036 — Fine-Tuned Hybrid Evidence Cross-Encoder

### Hypothesis

Expanding the EXP-M030 sentence candidate space with compact 6-word and 10-word windows may improve evidence localization while preserving a manageable inference candidate count.

### Controlled change

EXP-M030:

* 1–3 sentence candidates

EXP-M036:

* 1–3 sentence candidates
* 6-word windows
* 10-word windows
* word-window stride 3

All other major training settings were retained from EXP-M030:

* base model: `cross-encoder/ms-marco-MiniLM-L6-v2`
* 5-fold GroupKFold by conversation
* temporal-IoU supervision
* hard-negative sampling
* optimizer and learning rate
* epoch count
* batch configuration

### Results

195 gold-positive questions.

OOF localization:

* mean tIoU: 0.5144
* median tIoU: 0.5650
* any overlap: 0.7590
* tIoU >= 0.25: 0.7128
* tIoU >= 0.50: 0.5590
* tIoU >= 0.75: 0.3641

Compact hybrid candidate oracle:

* mean tIoU: 0.8447

Wall time:

* 508.6 seconds

### Comparison with EXP-M030

EXP-M030:

* mean tIoU: 0.5089
* median tIoU: 0.5274
* tIoU >= 0.50: 0.5128
* tIoU >= 0.75: 0.3692

EXP-M036:

* mean tIoU: 0.5144
* median tIoU: 0.5650
* tIoU >= 0.50: 0.5590
* tIoU >= 0.75: 0.3641

Mean improvement:

+0.0055

### Interpretation

The compact hybrid candidate space produces a measurable but modest improvement in mean OOF tIoU.

It substantially increases the proportion of questions reaching at least 0.50 tIoU, but does not improve the highest-quality localization tail.

The large gap between realized OOF performance (0.5144) and the compact candidate oracle (0.8447) indicates ranking, rather than candidate geometry, remains the dominant bottleneck.

### Decision

KEEP as an experimental improvement.

Do not replace the deployed system until the composite score with the frozen EXP-M018 classifier is measured.

---

## EXP-M037 — M036 Hybrid Composite Development Score

### Goal

Measure whether the higher candidate-space ceiling from EXP-M036 translates into a meaningful end-to-end score improvement with the frozen EXP-M018 classifier.

### Results

Classification:

* accuracy: 0.8821
* gold-positive questions predicted YES: 170/195

Evidence:

* M036 overall OOF mean tIoU: 0.5144
* mean scored tIoU: 0.4436
* tIoU among returned gold positives: 0.5089

Composite:

* 0.6190

### Comparison

EXP-M031 sentence-neural composite:

* 0.6185

EXP-M037 hybrid-neural composite:

* 0.6190

Improvement:

* +0.0005

### Interpretation

The compact hybrid candidate space provides a small localization improvement but almost none of it translates into the final scored positive subset.

The additional inference burden is therefore not justified for deployment at this stage.

The large remaining oracle gap indicates that ranking quality, rather than candidate geometry, is the dominant evidence-localization bottleneck.

### Decision

Do not deploy EXP-M036.

Keep EXP-M030 / EXP-M031 as the current evidence reference.

Next experiment: improve the neural ranking objective directly.

---

## EXP-M038 — MiniLM-L12 Hybrid Cross-Encoder

Hypothesis:
Increasing cross-encoder capacity from MiniLM-L6 to MiniLM-L12 may improve ranking within the compact hybrid candidate space.

Change made:
- Base model: cross-encoder/ms-marco-MiniLM-L12-v2
- Same sentence + 6/10-word candidate geometry as M036
- Same 5-fold conversation-disjoint validation
- Batch size reduced for GTX 1060 6 GB

Results:
- Mean tIoU: 0.4977
- Median tIoU: 0.5526
- Any overlap: 0.7385
- tIoU >= 0.50: 0.5436
- tIoU >= 0.75: 0.3487
- Candidate oracle: 0.8447
- Wall time: 1021.1 s

Reference:
- M030 L6 sentence: 0.5089
- M036 L6 hybrid: 0.5144

Interpretation:
The larger L12 cross-encoder is both slower and worse than the L6 hybrid model. Increased model capacity does not improve this task under the current training objective.

Decision:
DISCARD.

---

## EXP-M041 — Four-Seed Evidence Ensemble

### Hypothesis

Averaging independently fine-tuned M036 models across four random seeds may reduce ranking variance and outperform any individual seed.

### Models

Same architecture and training configuration:

* `cross-encoder/ms-marco-MiniLM-L6-v2`
* compact hybrid sentence + 6-word + 10-word candidates
* 4 epochs
* learning rate 2e-5

Seeds:

* 42
* 1337
* 2026
* 31415

### Individual OOF results

* seed 42: 0.5144
* seed 1337: 0.5018
* seed 2026: 0.5179
* seed 31415: 0.5067

### Ensemble results

Equal raw-score mean:

* mean tIoU: 0.5096
* median tIoU: 0.5610
* any overlap: 0.7641
* tIoU >= 0.50: 0.5487
* tIoU >= 0.75: 0.3538

Within-question rank mean:

* mean tIoU: 0.5099

Median-score ensemble:

* mean tIoU: 0.5095

### Composite

Using the frozen M018 classifier:

* accuracy: 0.8821
* mean scored tIoU: 0.4388
* composite: 0.6161

Reference M037:

* composite: 0.6190

### Interpretation

The independent seed models do not combine beneficially through simple score averaging.

Averaging appears to smooth useful high-confidence candidate distinctions rather than reduce harmful variance.

The best individual seed is 2026 at 0.5179, but choosing it after observing OOF performance would introduce seed-selection optimism.

### Decision

DISCARD seed ensemble.

Retain M036/M037 as the clean development reference.


---

## EXP-M042 — Composite-Aware Meta-Classifier

### Hypothesis

A supervised meta-classifier combining multiple NLI confidence signals can recover additional gold-positive questions and improve the competition composite score, even if raw accuracy decreases slightly.

### Validation

5-fold GroupKFold grouped by `transcript_id`.

Decision thresholds were calibrated only on training conversations in each fold.

Threshold optimization used the actual competition objective:

`0.4 × Accuracy + 0.6 × mean positive tIoU`

Gold labels, question type, correctness, and evidence annotations were not used as inference-time model features.

### Logistic regression

* accuracy: 0.8410
* mean scored tIoU: 0.4646
* composite: 0.6152
* TP / FN: 175 / 20
* TN / FP: 153 / 42

Decision: DISCARD.

### HistGradientBoosting classifier

* accuracy: 0.8744
* mean scored tIoU: 0.4665
* composite: 0.6297
* predicted YES rate: 0.5333
* TP / FN: 177 / 18
* TN / FP: 164 / 31

Fold composites:

* fold 1: 0.6339
* fold 2: 0.7119
* fold 3: 0.6234
* fold 4: 0.6335
* fold 5: 0.5347

Reference EXP-M037:

* accuracy: 0.8821
* mean scored tIoU: 0.4436
* composite: 0.6190

Composite improvement:

+0.0107

### Interpretation

The HGB meta-classifier improves the competition objective by recovering seven additional gold-positive questions.

The score improvement is large enough to keep, but one held-out fold performs substantially worse than the others.

Because the feature set includes question-surface variables such as auxiliary verb, length, number presence, and negation, a targeted ablation is required before deployment to determine whether the gain depends on potentially brittle question-template shortcuts.

### Decision

KEEP as experimental best classifier.

Do not deploy until NLI-only versus question-surface feature ablation is complete.

---

## EXP-M043 — Meta-Classifier Feature Ablation

### Goal

Determine whether the EXP-M042 improvement depends primarily on brittle question-template features or survives with more general NLI-derived signals.

### Results

#### NLI only

* accuracy: 0.8667
* mean scored tIoU: 0.4575
* composite: 0.6212

#### NLI + semantic surface features

Features additionally included question length, explicit number/unit presence, and negation.

* accuracy: 0.8744
* mean scored tIoU: 0.4592
* composite: 0.6252

#### Full EXP-M042

Also included question auxiliary-form indicators.

* accuracy: 0.8744
* mean scored tIoU: 0.4665
* composite: 0.6297

### Reference

EXP-M037:

* accuracy: 0.8821
* mean scored tIoU: 0.4436
* composite: 0.6190

### Interpretation

The EXP-M042 improvement is not entirely attributable to question-template features.

NLI-only meta-classification already slightly exceeds the previous system, and adding general surface features increases the gain further.

Auxiliary-form indicators provide an additional approximately +0.0044 composite, so some dependence on dataset question structure remains, but they are not the sole source of the improvement.

### Decision

KEEP full EXP-M042 as the current best development classifier.

Next: improve threshold calibration robustness using nested conversation-disjoint cross-validation.

---

## EXP-M044 — Nested Threshold Calibration

### Goal

Test whether nested conversation-disjoint threshold calibration improves robustness of the meta-classifier and reduces threshold overfitting.

### Method

Outer 5-fold GroupKFold by conversation.

Within each outer training partition:

* inner 4-fold GroupKFold
* generate inner OOF probabilities
* optimize the decision threshold against the competition composite objective
* retrain the classifier on all outer-training conversations
* apply the calibrated threshold to the outer held-out fold

This experiment used NLI-only inference features.

### Results

* accuracy: 0.8487
* mean scored tIoU: 0.4619
* composite: 0.6166
* TP / FN: 178 / 17
* TN / FP: 153 / 42

Fold-calibrated thresholds:

* 0.2487
* 0.2595
* 0.3569
* 0.2215
* 0.3690

### References

EXP-M037:

* composite: 0.6190

EXP-M043 NLI-only:

* composite: 0.6212

EXP-M042 full:

* composite: 0.6297

### Interpretation

Nested calibration slightly improved positive recall but substantially increased false positives.

The loss in accuracy outweighed the evidence benefit.

### Decision

**DISCARD.**

Retain EXP-M042 full HGB as the stronger meta-classifier.

---

## EXP-M045 — M042 Differential Error Analysis

### Goal

Understand exactly how M042 differs from the previous M018 classifier and determine whether its additional errors can be corrected systematically.

### Comparison

M018 errors:

* 46

M042 errors:

* 49

Among M018/M042 disagreements:

* M042 fixed M018 errors: 8
* M042 introduced new errors: 11
* wrong in both systems: 38

The systems disagreed on only 19 of 390 questions.

### Key Pattern

Most useful M042 changes were:

* M018 = NO
* M042 = YES
* gold = YES

Most harmful M042 changes were also:

* M018 = NO
* M042 = YES
* gold = NO

Therefore M042's main benefit and main failure mode were both concentrated in the same type of override.

Examples of useful M042 rescues included:

* Ibumetin prescription issued
* continued monitoring
* fasting blood sugar of 7.0 mmol/L
* HbA1c of 43 mmol/mol
* no reaction after injection
* cholesterol level requested

Examples of harmful M042 overrides included:

* off-topic pet/device questions
* hospital admission
* insulin initiation
* wounds on feet
* severe obesity
* antibiotic treatment

### Interpretation

M042 improved positive recall, but its additional YES predictions were not uniformly trustworthy.

Simple rules based on M042 probability or original NLI margin did not clearly separate useful from harmful overrides.

### Decision

**KEEP M042, but investigate disagreement-specific evidence confidence rather than applying hand-written correction rules.**

---

## EXP-M046 — Evidence-Aware Positive Disagreement Audit

### Goal

Test whether the final evidence ranker provides an independent signal for deciding whether M042 should override M018.

### Scope

At this stage, final OOF evidence spans were available only for the 195 gold-positive questions.

This allowed analysis of:

* 8 useful M042 overrides
* 1 harmful positive-direction change

### Results

For M042 changes that fixed M018:

* mean evidence-ranker score: 0.7328
* mean evidence-NLI entailment: 0.2202
* mean evidence-NLI contradiction: 0.1697
* mean evidence-NLI ratio: 0.4516
* mean evidence gold tIoU: 0.5585

For the one harmful positive change:

* evidence-ranker score: 0.4069
* evidence-NLI entailment: 0.0013
* evidence-NLI contradiction: 0.9761
* evidence-NLI ratio: 0.0014
* evidence gold tIoU: 0.0000

### Interpretation

The final evidence-ranker confidence appeared much more useful than the classifier's internal NLI segment scores.

However, the harmful comparison group contained only one positive example, so no gating rule could yet be considered reliable.

### Decision

**KEEP as supporting analysis.**

Next step: generate inference-realistic evidence spans for all 390 questions.

---

## EXP-M047 — All-Question Hybrid OOF Evidence Ranker

### Goal

Extend the M036 hybrid evidence ranker to all 390 questions so that evidence confidence can be analyzed for both positive and negative classifier disagreements.

### Method

Candidate types:

* sentence_1
* sentence_2
* sentence_3
* word_6
* word_10

Training supervision:

* positive questions only

Held-out scoring:

* all positive questions
* all hard negatives
* all off-topic questions

Five-fold GroupKFold by conversation was preserved.

### Coverage

* all questions: 390
* positive: 195
* hard_negative: 142
* off_topic: 53

### Positive Localization

* mean tIoU: 0.5158
* median tIoU: 0.5673
* any overlap: 0.7641
* tIoU >= 0.25: 0.7077
* tIoU >= 0.50: 0.5744
* tIoU >= 0.75: 0.3641

Candidate oracle:

* mean tIoU: 0.8447

Runtime:

* wall time: 537.9 s

### Reference

M036 positive-only evidence:

* mean tIoU: 0.5144

### Interpretation

The all-question version preserved evidence quality and slightly improved positive localization.

More importantly, it produced an inference-realistic selected evidence span and ranker confidence for every one of the 390 questions.

### Decision

**KEEP.**

M047 becomes the evidence source for disagreement arbitration analysis.

---

## EXP-M048 — All-Disagreement Evidence Audit

### Goal

Analyze all 19 M018/M042 disagreements using the M047 final evidence span and evidence confidence.

### Disagreement Structure

* total disagreements: 19
* M042 useful changes: 8
* M042 harmful changes: 11

Direction:

* NO → YES: 18
* YES → NO: 1

### Group Means

#### M042 fixed M018

* M042 probability: 0.5299
* M042 margin above threshold: 0.1529
* max segment ratio: 0.3948
* max segment entailment: 0.0647
* max segment margin: -0.000934
* evidence-ranker score: **0.7001**
* evidence-NLI entailment: 0.2165
* evidence-NLI contradiction: 0.1964
* evidence-NLI ratio: 0.4257

#### M042 broke M018

* M042 probability: 0.4938
* M042 margin above threshold: 0.1192
* max segment ratio: 0.4369
* max segment entailment: 0.0028
* max segment margin: -0.000920
* evidence-ranker score: **0.2342**
* evidence-NLI entailment: 0.0012
* evidence-NLI contradiction: 0.1021
* evidence-NLI ratio: 0.1336

### Separation

Evidence-ranker score:

* useful range: 0.3888–0.9559
* harmful range: 0.0511–0.5681
* useful mean: 0.7001
* harmful mean: 0.2342

Original max-segment NLI margin showed essentially no separation:

* useful mean: -0.000934
* harmful mean: -0.000920

### Interpretation

The final evidence ranker's confidence was substantially more informative for arbitration than the original classification NLI signals.

A large fraction of harmful M042 YES overrides had weak evidence-ranker confidence.

### Decision

**KEEP.**

Proceed to a conversation-disjoint evidence-confidence gate between M018 and M042.

---

## EXP-M049 — Evidence-Gated M018/M042 Arbitration

### Goal

Preserve the useful positive-recall gains from M042 while rejecting the additional false positives it introduced.

### Motivation

M018 and M042 disagreed on only 19 of 390 development questions.

Among those disagreements:

* M042 fixed 8 M018 errors
* M042 introduced 11 new errors

M048 showed that final evidence-ranker confidence strongly separated many useful M042 overrides from harmful ones.

### Method

M018 was retained as the conservative base classifier.

For each question:

1. Generate the M018 prediction.
2. Generate the M042 HGB meta-classifier prediction.
3. If both classifiers agree, use their shared answer.
4. If they disagree:

   * trust M042 when final evidence-ranker confidence exceeds a calibrated threshold
   * otherwise retain M018.

Thresholds were calibrated within the training portion of each conversation-disjoint GroupKFold split.

### Fold Results

Fold 1:

* threshold: 0.3398
* test disagreements: 2
* M042 overrides accepted: 2
* accuracy: 0.9250
* scored tIoU: 0.4180
* composite: 0.6208

Fold 2:

* threshold: 0.6215
* test disagreements: 7
* M042 overrides accepted: 3
* accuracy: 0.9250
* scored tIoU: 0.5414
* composite: 0.6948

Fold 3:

* threshold: 0.3693
* test disagreements: 3
* M042 overrides accepted: 1
* accuracy: 0.8625
* scored tIoU: 0.4828
* composite: 0.6347

Fold 4:

* threshold: 0.3693
* test disagreements: 4
* M042 overrides accepted: 2
* accuracy: 0.8625
* scored tIoU: 0.5032
* composite: 0.6469

Fold 5:

* threshold: 0.3693
* test disagreements: 3
* M042 overrides accepted: 1
* accuracy: 0.8714
* scored tIoU: 0.3665
* composite: 0.5684

### Overall OOF Result

* accuracy: 0.8897
* mean scored tIoU: 0.4658
* composite: **0.6354**
* TP / FN: 176 / 19
* TN / FP: 171 / 24

Disagreements:

* total: 19
* accepted M042 overrides: 9
* rejected M042 overrides: 10

Fold thresholds:

* 0.3398
* 0.6215
* 0.3693
* 0.3693
* 0.3693

Mean threshold:

* 0.4138

Median threshold:

* approximately 0.3693

### References

M037 / M018:

* composite: 0.6190

M042:

* composite: 0.6297

M049:

* composite: **0.6354**

### Interpretation

Evidence-ranker confidence provides useful independent information for choosing between M018 and M042.

M049 improves both classification accuracy and the competition composite relative to M042 by retaining useful positive rescues while rejecting many low-confidence false-positive overrides.

### Decision

**KEEP — CURRENT BEST CONVERSATION-DISJOINT SYSTEM.**

---

## EXP-M050 — Fixed M049 Deployment Threshold Diagnostic

### Goal

Determine whether a single threshold suitable for deployment can approximate or improve upon the fold-specific M049 arbitration thresholds.

### Motivation

Three of the five M049 folds independently selected a threshold near:

`0.3693`

A single deployment threshold simplifies the runtime system and avoids requiring fold-dependent logic.

### Threshold

`0.369304`

### Result

* accuracy: **0.8974**
* mean scored tIoU: **0.4732**
* composite: **0.6429**
* TP / FN: 178 / 17
* TN / FP: 172 / 23
* accepted M042 overrides: 10

### Comparison

M049 unbiased OOF:

* composite: 0.6354

M050 fixed-threshold diagnostic:

* composite: 0.6429

### Important Validation Caveat

The threshold was selected after examining the cross-validation threshold behavior.

Therefore the M050 value is **not an unbiased OOF estimate**.

The defensible development estimate remains:

* M049 OOF composite: **0.6354**

### Interpretation

A fixed threshold near 0.3693 appears suitable for deployment and produces a strong development diagnostic.

### Decision

**KEEP as deployment configuration.**

Use:

`EVIDENCE_GATE_THRESHOLD = 0.369304`

Do not report 0.6429 as an unbiased validation result.

---

## EXP-M049 Deployment Export and Integration Validation

### Goal

Train the final M049 components on all supplied development data and integrate them into the production `/predict` endpoint.

### Exported Models

M042 meta-classifier:

`medical/artifacts/models/meta_classifier_m042.joblib`

Hybrid evidence ranker:

`medical/artifacts/models/hybrid_evidence_ranker_m049/`

Deployment metadata:

`medical/artifacts/models/m049_deployment_metadata.json`

### M042 Deployment Model

Training rows:

* 390

Features:

* 22

Model:

* HistGradientBoostingClassifier

Mean training probability of YES:

* 0.5002

### Hybrid Evidence Ranker

Training supervision:

* 195 positive questions

Candidate count:

* 70,724

Selected training pairs:

* 7,545

Training losses:

* epoch 1: 0.064611
* epoch 2: 0.035311
* epoch 3: 0.025862
* epoch 4: 0.021242

### Runtime Thresholds

M018 classification threshold:

`0.000585`

M042 probability threshold:

`0.384878`

M049 evidence gate:

`0.369304`

### Endpoint Smoke Test

* HTTP: 200
* structure: PASS
* predicted YES: 4/10
* accuracy: 9/10
* latency: 7.23 s

### Full Local Integration Evaluation

Questions:

* 390

Correct:

* 355

Failed conversations:

* 0

Timeouts:

* 0

Accuracy by type:

* positive: 180/195 = 0.923
* hard_negative: 124/142 = 0.873
* off_topic: 51/53 = 0.962

Evidence:

* mean tIoU: 0.695
* no span returned: 15
* tIoU when answered YES: 0.752

Latency:

* mean conversation: 7.016 s
* worst conversation: 12.838 s
* evaluator budget: 60 s/conversation

Final local result:

* accuracy: **0.910**
* mean tIoU: **0.695**
* composite: **0.781**

### Important Caveat

The final deployment models were trained on all 390 supplied development questions.

Therefore the 0.781 local score is an integration/training-set diagnostic and must not be treated as a generalization estimate.

The primary unbiased development reference remains:

* M049 OOF: 0.6354

### Official Hidden Validation

Previous deployed system:

* hidden validation score: 0.584396

M049 deployment:

* hidden validation score: **0.617839**

Absolute improvement:

* **+0.033443**

Relative improvement:

* approximately **+5.7%**

Errors:

* none

### Interpretation

The hidden score confirms that the M049 architecture genuinely improves generalization relative to the previous deployed system.

The hidden score is also reasonably close to the conversation-disjoint OOF estimate:

* OOF: 0.6354
* hidden: 0.6178

### Decision

**KEEP M049 as the current production baseline.**

---

## EXP-M051 — Top-5 HGB Evidence Reranker

### Goal

Exploit the large difference between top-1 evidence performance and the oracle quality available among the top five M047 candidates.

### Motivation

M047 candidate recall analysis:

* top-1 oracle tIoU: 0.5158
* top-2 oracle: 0.5856
* top-3 oracle: 0.6302
* top-5 oracle: 0.6893
* top-10 oracle: 0.7461
* top-20 oracle: 0.7966
* full candidate oracle: 0.8447

This showed that the correct evidence was frequently already near the top of the ranking.

### Method

For each positive question:

* retain M047 top-5 candidates
* construct candidate-level inference-safe features
* train an HGB regressor to predict candidate tIoU
* select the candidate with the highest predicted score

Features included:

* first-stage evidence score
* first-stage rank
* score gaps
* duration
* word count
* sentence count
* lexical similarity
* MS-MARCO score
* token coverage
* Jaccard similarity
* DeBERTa entailment / contradiction / neutral
* NLI margin
* NLI ratio
* candidate-type indicators

Validation:

* 5-fold GroupKFold by conversation

### Fold Results

Fold 1:

* M047: 0.5107
* M051: 0.5106
* delta: -0.0001

Fold 2:

* M047: 0.5910
* M051: 0.5945
* delta: +0.0036

Fold 3:

* M047: 0.3579
* M051: 0.3960
* delta: +0.0381

Fold 4:

* M047: 0.6281
* M051: 0.6080
* delta: -0.0201

Fold 5:

* M047: 0.4816
* M051: 0.4287
* delta: -0.0528

### Overall Result

* mean tIoU: 0.5092
* median tIoU: 0.5505
* any overlap: 0.7692
* tIoU >= 0.25: 0.6974
* tIoU >= 0.50: 0.5436
* tIoU >= 0.75: 0.3641

Reference:

* M047: 0.5158
* top-5 oracle: 0.6893

### Interpretation

The top-5 candidate set contains substantial unused evidence quality, but a generic tabular regression model cannot reliably identify the best span.

Performance is unstable across folds and slightly worse than the original first-stage ranker overall.

### Decision

**DISCARD.**

---

## EXP-M052 — Pairwise Top-5 Tabular Reranker

### Goal

Test whether pairwise preference learning can exploit the strong M047 top-5 candidate recall more effectively than absolute tIoU regression.

### Method

For each positive question:

* retain the M047 top five candidates
* generate every non-tied candidate pair
* label the candidate with higher gold tIoU as preferred
* add reversed pairs to maintain class symmetry

Feature vectors were candidate-feature differences derived from the M051 feature set.

Model:

* standardized logistic regression

Inference:

* round-robin pairwise comparisons
* candidate with most wins selected
* aggregate pairwise confidence used for tie breaking

Validation:

* conversation-disjoint 5-fold GroupKFold

### Fold Results

Fold 1:

* training pairs: 2,574
* M047: 0.5107
* M052: 0.4826
* delta: -0.0281

Fold 2:

* training pairs: 2,504
* M047: 0.5910
* M052: 0.5561
* delta: -0.0348

Fold 3:

* training pairs: 2,578
* M047: 0.3579
* M052: 0.4035
* delta: +0.0457

Fold 4:

* training pairs: 2,560
* M047: 0.6281
* M052: 0.6190
* delta: -0.0092

Fold 5:

* training pairs: 2,600
* M047: 0.4816
* M052: 0.4870
* delta: +0.0054

### Overall Result

* mean tIoU: 0.5108
* median tIoU: 0.5630
* any overlap: 0.7179
* tIoU >= 0.25: 0.6821
* tIoU >= 0.50: 0.5590
* tIoU >= 0.75: 0.4000

Reference:

* M047 top-1: 0.5158
* M051 HGB: 0.5092
* top-5 oracle: 0.6893

Gain versus M047:

* -0.0050

### Interpretation

Pairwise learning does not solve the top-5 selection problem when based on handcrafted candidate features.

Both M051 and M052 fail despite a strong top-5 oracle.

This indicates that the unresolved distinction is likely semantic and should be modeled directly from candidate text rather than through engineered feature differences.

### Decision

**DISCARD.**

Do not continue with additional generic tabular rerankers.

---

## EXP-M053 — Challenge Structure and Runtime-Signal Audit

### Goal

Investigate whether leaderboard scores near 1.00 can be explained by deterministic challenge structure, question ordering, repeated templates, question wording, or underused runtime-safe model signals.

### Dataset Structure

* questions: 390
* conversations: 39
* questions per conversation: exactly 10
* YES: 195
* NO: 195

Question types:

* positive: 195
* hard_negative: 142
* off_topic: 53

YES counts per conversation varied substantially:

* 3 YES: 7 conversations
* 4 YES: 10
* 5 YES: 7
* 6 YES: 6
* 7 YES: 9

Every observed question-type ordering was effectively unique.

### Question Position

YES rates varied by position but were not strongly deterministic.

Position-only diagnostic:

* accuracy: 0.5897
* TN / FP: 115 / 80
* FN / TP: 80 / 115

Conclusion:

Question ordering does not provide a strong shortcut.

### Question Opening

Some question openings were imbalanced, e.g. questions beginning with `the` or `will`, but the common auxiliary forms were not deterministic enough to explain high leaderboard performance.

### Duplicate / Template Analysis

* exact duplicate groups: 3
* normalized duplicate groups: 6
* normalized duplicate groups with mixed labels: 2

Repeated templates were rare.

### Question-Only Classification

Character TF-IDF + logistic regression, conversation-disjoint:

* accuracy: 0.7256
* TN / FP: 140 / 55
* FN / TP: 52 / 143

Random question-level CV:

* accuracy: 0.6564
* TN / FP: 123 / 72
* FN / TP: 62 / 133

Conclusion:

Question text carries some label information but no obvious high-performing template leak.

Random question CV does not outperform conversation-disjoint CV, providing no evidence that repeated templates explain near-perfect scores.

### Runtime-Safe Numeric Signals

Features:

* question position
* max segment ratio
* max segment entailment
* max segment margin
* max segment vs other
* selected entailment
* selected contradiction
* selected neutral
* maximum evidence-ranker score
* mean evidence-ranker score
* evidence-ranker score standard deviation
* top-1 / top-2 evidence score gap

Conversation-disjoint HGB result:

* accuracy: **0.9026**
* TN / FP: 173 / 22
* FN / TP: 16 / 179

Total errors:

* **38 / 390**

This matches the previously measured 38-error oracle ceiling obtained by choosing perfectly between M018 and M042.

### Question Text + Runtime Signals

Combined character TF-IDF question representation plus numeric runtime signals:

* accuracy: 0.8949
* TN / FP: 174 / 21
* FN / TP: 20 / 175

The text representation reduced performance relative to numeric signals alone.

### Interpretation

No simple positional, duplicate-template, or question-text leakage explains the task.

The strongest new finding is instead that evidence-ranker score-distribution statistics contain substantial classification information.

A conversation-disjoint classifier using only inference-safe numeric signals reaches 90.26% accuracy, outperforming prior individual classifiers and matching the previously observed M018/M042 oracle error count.

Question text should not be added to this classifier under the current formulation.

### Decision

**KEEP — major finding.**

Next experiment: construct a composite-aware classifier from these runtime-safe numeric signals and evaluate its interaction with M049.

---

## EXP-M054 — Runtime-Signal Composite Classifier

### Goal

Convert the strong runtime-safe numeric signal discovered in M053 into a classifier optimized for the actual competition composite objective.

### Features

Inference-safe features only:

* question position
* max segment ratio
* max segment entailment
* max segment margin
* max segment vs other
* selected entailment
* selected contradiction
* selected neutral
* maximum evidence-ranker score
* mean evidence-ranker score
* evidence-ranker score standard deviation
* top-1 / top-2 evidence score gap

No gold labels, question type, or evidence annotations were used as inference features.

### Validation

5-fold GroupKFold by conversation.

Each fold:

1. trained HGB on outer-training conversations
2. selected a composite-optimal probability threshold on the training partition
3. applied that threshold to the held-out conversations

### OOF Result

* accuracy: **0.9000**
* scored tIoU: **0.4806**
* composite: **0.6484**
* TP / FN: 179 / 16
* TN / FP: 172 / 23

Fold thresholds:

* 0.4970
* 0.5344
* 0.5433
* 0.5284
* 0.4706

Mean threshold:

* 0.5147

### References

M053 default runtime-numeric classifier:

* accuracy: 0.9026

M049:

* OOF composite: 0.6354
* hidden validation: 0.6178

M054:

* OOF composite: **0.6484**

Gain over M049 OOF:

* **+0.0130**

### Interpretation

Runtime evidence-score distribution features carry substantial classification information beyond the original segmentwise NLI classifier.

M054 improves the competition objective while maintaining approximately 90% classification accuracy.

Threshold selection is comparatively stable across folds, clustering around roughly 0.47–0.54.

### Decision

**KEEP — CURRENT BEST CLEAN OFFLINE SYSTEM.**

Next: evaluate complementarity between M054 and M049/M018/M042 and determine whether an arbitration ensemble can exceed 0.6484.

## EXP-M055A — M049 / M054 Complementarity Audit

### Goal

Measure whether M049 and M054 make sufficiently different errors to justify classifier arbitration.

### Results

- M054 errors: 39
- M049 errors: 43
- errors shared by both: 28
- M054 fixes M049: 15
- M049 fixes M054: 11

Oracle choosing correctly between M049 and M054:

- errors: 28
- accuracy: 0.9282

### Interpretation

M049 and M054 are substantially complementary.

M054 is stronger overall, but M049 correctly handles 11 examples that M054 misses. Conversely, M054 fixes 15 M049 errors.

This provides meaningful headroom beyond the standalone M054 classifier.

### Decision

KEEP as a major ensemble opportunity.

Next: analyze disagreement confidence and construct a conversation-disjoint arbitration rule.

---

## EXP-M055 — DeBERTa-v3-base Runtime-Signal Classifier

### Goal

Test whether a stronger NLI model improves the runtime-safe numeric classification signals discovered in M053/M054.

### NLI Model

`cross-encoder/nli-deberta-v3-base`

The model successfully loaded on the GTX 1060 6 GB.

### Raw Segmentwise NLI Result

* accuracy: 0.7641
* predicted YES rate: 0.7000
* gold YES rate: 0.5000

By question type:

* positive: 188/195 = 0.9641
* hard_negative: 73/142 = 0.5141
* off_topic: 37/53 = 0.6981

Confusion matrix:

* TN: 110
* FP: 85
* FN: 7
* TP: 188

Latency:

* total for 390 questions: 22.68 s
* mean/question: 0.0581 s
* estimated/10 questions: 0.5814 s

### Interpretation of Raw NLI

The larger NLI model strongly favors YES under the existing segmentwise decision rule.

Its raw classification accuracy is substantially worse than the smaller NLI model because of excessive false positives.

However, its probability distributions may still provide stronger semantic features.

### Runtime-Signal Classifier

The same M054 runtime-safe feature architecture was rebuilt using the DeBERTa-v3-base NLI features.

Features included:

* question position
* max segment ratio
* max segment entailment
* max segment margin
* max segment vs other
* selected entailment
* selected contradiction
* selected neutral
* evidence maximum score
* evidence mean score
* evidence score standard deviation
* evidence top-1 / top-2 gap

Validation:

* 5-fold GroupKFold by conversation
* fold-local composite threshold calibration

### OOF Result

* accuracy: **0.9128**
* scored tIoU: **0.4805**
* composite: **0.6534**
* TP / FN: 174 / 21
* TN / FP: 182 / 13

Fold thresholds:

* 0.5083
* 0.4353
* 0.5649
* 0.4919
* 0.4942

Mean threshold:

* 0.4989

### References

M054:

* accuracy: 0.9000
* composite: 0.6484

M055:

* accuracy: **0.9128**
* composite: **0.6534**

Gain over M054:

* composite: **+0.0050**
* accuracy: **+0.0128**

### Interpretation

Although DeBERTa-v3-base performs poorly as a direct segmentwise classifier, its NLI probability landscape is more useful as input to the downstream HGB runtime-signal classifier.

The strongest improvement is in negative rejection:

* M054 false positives: 23
* M055 false positives: 13

This comes at the cost of additional false negatives:

* M054 false negatives: 16
* M055 false negatives: 21

The net effect is still positive for the competition composite.

### Decision

**KEEP — CURRENT BEST CLEAN OOF SYSTEM.**

Do not use the raw DeBERTa-v3-base prediction directly.

Use its NLI outputs as features in the runtime-signal classifier.

---

## EXP-M056 — Dual-NLI Runtime-Signal Fusion

### Goal

Exploit the strong complementarity between M054 and M055 by combining the probability landscapes from both the small and base DeBERTa NLI models in a single inference-safe classifier.

### Motivation

M054:

* errors: 39

M055:

* errors: 34

M054/M055 shared errors:

* 14

Oracle choosing correctly between M054 and M055:

* accuracy: 0.9641

Therefore the two NLI systems contain substantially complementary classification information.

### Method

The fusion classifier used:

* all M054 small-DeBERTa NLI features
* all M055 base-DeBERTa NLI features
* evidence-ranker score-distribution features
* M054 probability
* M055 probability
* each model's signed distance from its calibrated threshold
* cross-model probability difference
* absolute probability difference

Model:

* HistGradientBoostingClassifier

Validation:

* conversation-disjoint 5-fold GroupKFold
* fold-local composite threshold calibration

### OOF Result

* accuracy: **0.9103**
* scored tIoU: **0.4825**
* composite: **0.6536**
* TP / FN: 179 / 16
* TN / FP: 176 / 19

Fold thresholds:

* 0.4679
* 0.4733
* 0.5272
* 0.4689
* 0.4908

Mean threshold:

* 0.4856

### References

M054:

* composite: 0.6484

M055:

* composite: 0.6534

M056:

* composite: **0.6536**

M054/M055 oracle classification accuracy:

* 0.9641

### Interpretation

Joint use of the two NLI probability landscapes improves positive recall relative to M055 while retaining much of M055's improved negative rejection.

However, the composite gain over M055 is only:

* **+0.0002**

This is effectively a tie given the dataset size.

The generic fusion model captures only a small fraction of the very large M054/M055 oracle headroom.

### Decision

**KEEP as current highest OOF composite, but do not prioritize deployment yet.**

Next: study the M054/M055 disagreement cases directly and determine whether an explicit disagreement arbiter can exploit their complementarity better than broad feature fusion.

---

## EXP-M057 — Intra-Conversation Question-Set Structure Audit

### Goal

Determine whether the ten questions submitted together for each conversation contain exploitable semantic relationships that independent per-question classification ignores.

### Pair Coverage

* conversations: 39
* within-conversation question pairs: 1,755

### Similarity vs Label Relationship

Character similarity >= 0.30:

* pairs: 45
* opposite labels: 0.8444
* same labels: 0.1556

Character similarity >= 0.40:

* pairs: 24
* opposite labels: 0.9583
* same labels: 0.0417

Character similarity >= 0.50:

* pairs: 10
* opposite labels: 1.0000
* same labels: 0.0000

Character similarity >= 0.60:

* pairs: 4
* opposite labels: 1.0000

Character similarity >= 0.70:

* pairs: 2
* opposite labels: 1.0000

Character similarity >= 0.80:

* pairs: 1
* opposite labels: 1.0000

### Examples

Observed high-similarity opposite-label pairs include:

* LDL cholesterol 4.2 mmol/L vs 2.2 mmol/L
* fasting glucose 5.0 mmol/L vs 7.0 mmol/L
* known COVID exposure vs explicitly denying exposure
* erythema migrans present vs absent
* continuing medication vs stopping medication
* HbA1c 43 vs 53 mmol/mol
* blood pressure 135/88 vs 155/98
* coatings present vs free of coatings
* diabetes unstable vs stable
* foreign body present vs ruled out
* pus present vs explicitly absent

### Interpretation

The challenge generator frequently creates hard-negative questions by modifying a positive proposition within the same conversation.

The relationship may involve:

* numeric substitution
* polarity reversal
* negation
* mutually exclusive diagnoses
* present vs absent findings
* continue vs stop treatment
* normal vs abnormal state

This structure is available at inference because all ten questions for a conversation arrive together in one request.

Independent classification therefore discards useful information.

### Decision

**KEEP — major structural finding.**

Next experiment: joint set-level consistency correction using highly similar question pairs.

---

## EXP-M058 — Pair-Consistency Correction

### Goal

Test whether high-similarity intra-conversation question pairs can be used as logical constraints to improve classification.

### Method

Baseline:

* M055 predictions

For every within-conversation pair above a character-similarity threshold:

* if predictions already differed, leave them unchanged
* if predictions were identical, preserve the higher-confidence prediction
* flip the lower-confidence prediction to enforce opposite labels

Thresholds tested:

* 0.80
* 0.70
* 0.60
* 0.50
* 0.45
* 0.40
* 0.35
* 0.30

### Baseline

M055:

* accuracy: 0.9128
* scored tIoU: 0.4805
* composite: 0.6534

### Results

Similarity >= 0.80:

* corrections: 0
* composite: 0.6534

Similarity >= 0.70:

* corrections: 0
* composite: 0.6534

Similarity >= 0.60:

* corrections: 1
* accuracy: 0.9154
* scored tIoU: 0.4805
* composite: 0.6544

Similarity >= 0.50:

* corrections: 1
* accuracy: 0.9154
* scored tIoU: 0.4805
* composite: 0.6544

Similarity >= 0.45:

* corrections: 2
* accuracy: 0.9128
* scored tIoU: 0.4770
* composite: 0.6513

Similarity >= 0.40:

* corrections: 5
* accuracy: 0.9205
* scored tIoU: 0.4770
* composite: 0.6544

Similarity >= 0.35:

* corrections: 10
* composite: 0.6489

Similarity >= 0.30:

* corrections: 16
* composite: 0.6453

### Interpretation

High-similarity question pairs contain useful structural information.

At strict thresholds, pairwise consistency can correct classification errors without harming evidence localization.

However, raw character similarity becomes too noisy as the threshold is relaxed.

A generic “similar questions must have opposite labels” rule is therefore insufficient.

### Decision

**KEEP as proof of concept, but do not deploy directly.**

Next experiment: explicitly detect semantic transformations such as numeric substitutions, polarity reversals, present/absent findings, continue/stop treatment, and normal/abnormal states.

---

## EXP-M060 — Precision Transformation Constraints

### Goal

Improve M058 by replacing generic character-similarity constraints with higher-precision semantic transformation detection.

### Method

Candidate intra-conversation pairs were retained only when they contained:

* high-confidence semantic opposites such as stable/unstable, normal/abnormal, continue/stop, viral/bacterial, deny/report, ruled-out/found
* or numeric substitutions where the surrounding question text remained highly similar after replacing numbers with placeholders

Noisy rules such as generic `free of ↔ has` were removed.

For detected pairs, if M055 predicted identical labels for both questions:

* retain the higher-confidence prediction
* flip the lower-confidence prediction

### Results

Detected candidate pairs:

* 21

M055 baseline:

* accuracy: 0.9128

---

## EXP-M061 — DeBERTa-v3-base Top-20 Evidence Reranking

### Goal

Test whether the stronger DeBERTa-v3-base NLI model can exploit the substantial evidence quality already present in the M047 top-20 candidate set.

### Motivation

M047 evidence selection:

* mean tIoU: 0.5158

Top-20 candidate oracle:

* mean tIoU: 0.7966
* median tIoU: 0.8934
* any overlap: 0.9692
* tIoU >= 0.50: 0.9026
* tIoU >= 0.75: 0.7231

This shows that the correct evidence span is very frequently already among the top twenty candidates.

### Oracle Candidate Types

* sentence_1: 107
* sentence_2: 28
* word_6: 28
* word_10: 18
* sentence_3: 4
* word_14: 4
* word_18: 3
* word_24: 2
* word_32: 1

### Standalone DeBERTa-v3-base Evidence Ranking

Entailment:

* mean tIoU: 0.3462

Margin:

* mean tIoU: 0.3454

Entailment ratio:

* mean tIoU: 0.3456

All three were substantially worse than M047.

### Conversation-Disjoint Fusion

Existing M047 OOF ranker scores were normalized within each question and fused with DeBERTa-v3-base NLI scores.

#### Entailment Fusion

Fold ranker weights:

* 0.85
* 0.90
* 1.00
* 0.90
* 0.90

OOF:

* mean tIoU: 0.5127
* median tIoU: 0.5533
* any overlap: 0.7795
* tIoU >= 0.50: 0.5590
* tIoU >= 0.75: 0.3538

Gain versus M047:

* -0.0031

#### Margin Fusion

* mean tIoU: 0.5014

#### Ratio Fusion

* mean tIoU: 0.5049

### Interpretation

The larger NLI model provides useful classification information but does not provide useful fine-grained evidence-span ranking.

The CV optimizer consistently assigns most or all weight to the existing M047 evidence ranker.

Generic entailment confidence does not correspond closely enough to temporal overlap quality among semantically related candidate spans.

### Decision

**DISCARD for evidence ranking.**

Retain DeBERTa-v3-base only as a classification feature generator.

The remaining evidence problem should be treated as a dedicated span-localization/ranking task rather than generic NLI.

---

## EXP-M062 — Temporal Prior and Earliest-Mention Evidence Selection

### Goal

Test whether catastrophic evidence-localization failures are caused by the evidence ranker selecting later summaries or repeated mentions instead of the earlier annotated occurrence.

### Motivation

M047 top-1 localization showed a strongly bimodal error distribution:

* 87/195 questions had center error <= 0.5 seconds
* 139/195 had center error <= 2 seconds
* 40/195 had center error >= 5 seconds and zero tIoU

Many catastrophic selections appeared semantically valid but occurred far from the annotated evidence.

### Method A — Temporal Score Penalty

The normalized M047 evidence score was penalized according to the candidate's relative position in the transcript.

The temporal penalty strength was selected independently within each conversation-disjoint training fold.

### Result

M047 baseline:

* mean tIoU: 0.5158

Temporal-prior folds:

* fold 1: -0.0115
* fold 2: +0.0000
* fold 3: +0.0000
* fold 4: +0.0000
* fold 5: +0.0000

OOF:

* mean tIoU: 0.5135
* median tIoU: 0.5673
* any overlap: 0.7590
* tIoU >= 0.50: 0.5744
* tIoU >= 0.75: 0.3641

### Method B — Near-Tie Earliest Mention

For candidates whose M047 scores were within a fold-calibrated delta of the top candidate, select the earliest candidate.

### Result

Fold deltas:

* fold 1: -0.0072
* fold 2: +0.0000
* fold 3: +0.0000
* fold 4: -0.0076
* fold 5: -0.0264

OOF:

* mean tIoU: 0.5076
* median tIoU: 0.5533
* any overlap: 0.7744
* tIoU >= 0.50: 0.5641
* tIoU >= 0.75: 0.3282

### Interpretation

The annotation does not follow a sufficiently strong global “earliest valid mention” rule.

Although some catastrophic errors correspond to later repeated or summary mentions, adding a generic temporal prior degrades evidence selection on held-out conversations.

The catastrophic-localization problem therefore requires semantic disambiguation between repeated mentions rather than a position heuristic.

### Decision

**DISCARD.**

Do not add an earlier-evidence prior to the production ranker.

---

## EXP-M063 — Hard-Negative Evidence Ranker

### Goal

Train the evidence cross-encoder specifically on difficult candidates that the existing M047 ranker scores highly but that have substantially worse gold tIoU.

### Method

Candidate pool:

* M047 top-20 candidates per positive question

Training pairs emphasized:

* candidate with strong gold overlap
* versus high-ranking candidates with substantially lower tIoU
* catastrophic zero-overlap candidates among M047's top ranks received additional training weight

Model:

* `cross-encoder/ms-marco-MiniLM-L6-v2`
* pairwise RankNet-style logistic objective
* conversation-disjoint 5-fold validation

### OOF Result

* mean tIoU: **0.5205**
* median tIoU: **0.6103**
* any overlap: 0.7436
* tIoU >= 0.25: 0.6974
* tIoU >= 0.50: 0.5795
* tIoU >= 0.75: 0.3744

Reference M047:

* mean tIoU: 0.5158

Gain:

* **+0.0047**

Selection behavior:

* selections changed: 72 / 195
* M047 zero-overlap failures rescued: 9
* new zero-overlap failures introduced: 13

Top-20 oracle:

* 0.7966

Remaining oracle gap:

* 0.2761

Wall time:

* 569.8 seconds

### Interpretation

Hard-negative mining improves mean and median evidence quality, confirming that training directly on M047's difficult confusions is more useful than generic NLI or tabular reranking.

However, M063 is too aggressive.

It repairs nine catastrophic M047 errors but introduces thirteen new zero-overlap selections, reducing overall overlap coverage.

The improvement therefore comes from better choices on a subset of cases while damaging others.

### Decision

**KEEP as a complementary evidence model.**

Do not replace M047 globally.

Next: measure M047/M063 complementarity and determine whether M063 can be gated to only high-confidence beneficial switches.

---

## EXP-M064 — M047/M063 Evidence Arbiter

### Goal

Exploit the complementarity between the original M047 evidence ranker and the hard-negative-trained M063 ranker without replacing M047 globally.

### Motivation

M047:

* mean tIoU: 0.5158

M063:

* mean tIoU: 0.5205

Two-model oracle:

* mean tIoU: 0.5771

Among 72 changed selections:

* M063 wins: 30
* M047 wins: 28
* ties: 14

### Models Tested

Three conversation-disjoint arbitration models:

* Ridge utility regression
* Logistic preference classification
* HGB utility regression

Features included:

* M063 rank within M047
* M047 and M063 score margins
* score gaps
* span durations
* word counts
* start/end/center displacement
* token overlap
* candidate kinds and kind transitions

The target was the utility of switching:

`tIoU(M063) - tIoU(M047)`

### Results

#### Ridge

* mean tIoU: 0.5122
* gain vs M047: -0.0036
* accepted M063: 51
* useful switches: 20
* harmful switches: 20

DISCARD.

#### Logistic

* mean tIoU: 0.5152
* gain vs M047: -0.0006
* accepted M063: 48
* useful switches: 19
* harmful switches: 18

DISCARD.

#### HGB

* mean tIoU: **0.5271**
* gain vs M047: **+0.0112**
* accepted M063: 35
* useful switches: 15
* harmful switches: 11
* neutral switches: 9

Fold deltas:

* fold 1: -0.0075
* fold 2: -0.0223
* fold 3: +0.0878
* fold 4: +0.0296
* fold 5: -0.0262

Two-model oracle:

* 0.5771

Oracle gap captured:

* 18.3%

### Interpretation

The M047 and M063 rankers contain useful complementary information, and a nonlinear arbiter can exploit some of it.

However, performance remains unstable across conversations. Tabular features do not reliably capture the semantic distinction between competing spans.

### Decision

**KEEP HGB M064 as current best evidence OOF model.**

Do not deploy yet.

Next: train a direct pairwise semantic comparator that jointly sees the question, the M047 candidate and the M063 candidate.

---

## EXP-M065 — Direct Pairwise Span Comparator

### Goal

Determine whether a transformer that jointly sees the question, the M047 candidate, and the M063 candidate can choose the better evidence span more reliably than the tabular M064 arbiter.

### Method

Model:

* `cross-encoder/ms-marco-MiniLM-L6-v2`

The original single-score MS-MARCO output head was replaced with a two-class classification head.

Training input:

* question
* candidate A
* candidate B

Target:

* which candidate has higher gold tIoU

Training pairs were derived from informative M047 top-5 candidate comparisons.

Validation:

* conversation-disjoint 5-fold cross-validation

### Result

* M047 mean tIoU: 0.5158
* M063 mean tIoU: 0.5205
* M064 HGB arbiter: 0.5271
* M065 mean tIoU: **0.5089**

Gain versus M047:

* **-0.0070**

Arbitration behavior:

* switched to M063: 47
* useful switches: 17
* harmful switches: 19
* neutral switches: 11

Two-model oracle:

* 0.5771

Oracle gap captured:

* -11.4%

Wall time:

* 189.5 seconds

### Interpretation

The direct binary-comparator formulation underperformed.

The original checkpoint was pretrained as a scalar query-passage relevance scorer. Replacing its pretrained ranking head with a randomly initialized two-class head discarded useful ranking structure and required the limited competition data to learn a substantially new comparison task.

The resulting comparator did not reliably distinguish beneficial from harmful M063 switches.

### Decision

**DISCARD.**

Do not use the two-class direct-comparator formulation again with this checkpoint.

---

## M066 — M063/M064 deployment evidence upgrade

**Goal:** Deploy the best evidence-side improvements without disturbing the existing M049 classification logic.

**Setup**
- Classification: existing M049 / M042 arbitration unchanged.
- Evidence:
  - M047/M049 first-stage evidence ranker.
  - M063 hard-negative evidence ranker.
  - M064 HGB arbiter deciding whether to keep the M047 span or switch to the M063 span.
- Important runtime decision: preserve the original M047 evidence confidence score for the M049 classification gate even when M064 changes the returned span.

**Local evaluator**
- Questions: 390
- Accuracy: **0.910**
- Mean tIoU: **0.715**
- Composite: **0.793**
- Failed conversations: 0
- Timeouts: 0
- Mean/conversation: 7.259 s
- Worst conversation: 13.210 s

Compared with the previous deployment:
- Previous local mean tIoU: ~0.695
- M066 local mean tIoU: **0.715**

**Hidden validation**
- Previous M049: **0.6178391635**
- M066: **0.6173885615**
- Delta: **-0.0004506**

**Conclusion:** DISCARDED.

The evidence improvements increased training/local-evaluator tIoU but did not transfer to hidden validation. This was the clearest indication so far that additional evidence-ranker optimization was overfitting the 39 provided conversations.

---

## M067 — M055 classification deployment

**Goal:** Replace the older classification path with the stronger M055 runtime-signal classifier while restoring the original pre-M066 evidence selector.

**Changes**
- NLI upgraded from:
  - `cross-encoder/nli-deberta-v3-small`
- To:
  - `cross-encoder/nli-deberta-v3-base`
- Classification model:
  - `meta_classifier_m055.joblib`
- Features:
  - `question_position`
  - `max_segment_ratio`
  - `max_segment_entailment`
  - `max_segment_margin`
  - `max_segment_vs_other`
  - `selected_entailment`
  - `selected_contradiction`
  - `selected_neutral`
  - `evidence_max_score`
  - `evidence_mean_score`
  - `evidence_std_score`
  - `evidence_top_gap`
- Evidence selection reverted to the original M049/M047 deployment selector.

**M055 OOF reference**
- Accuracy: **0.9128**
- Scored tIoU: **0.4805**
- Composite: **0.6534**
- TP/FN: 174 / 21
- TN/FP: 182 / 13

**Smoke test**
- HTTP 200
- Structure PASS
- Accuracy: 9/10
- Latency: 9.00 s

**Hidden validation**
- M049 baseline: **0.6178391635**
- M067: **0.6322466368**
- Delta vs M049: **+0.0144075**

**Conclusion:** ACCEPTED.

This was the first substantial hidden-validation improvement after the evidence experiments. Classification generalization was clearly more valuable than further evidence reranking.

---

## M068 — M054/M055 classification arbiter

**Goal:** Exploit the substantial disagreement complementarity between M054 and M055.

Earlier diagnostic:
- M054 errors: 39
- M055 errors: 34
- Both wrong: 14
- M054/M055 oracle accuracy: **0.9641**

A small meta-classifier was trained using both models' probabilities, margins, NLI features, evidence statistics, and question position.

### HGB OOF
- Accuracy: **0.9128**
- Scored tIoU: **0.4861**
- Composite: **0.6568**
- TP/FN: 183 / 12
- TN/FP: 173 / 22

### Logistic OOF
- Accuracy: **0.9026**
- Scored tIoU: **0.4918**
- Composite: **0.6561**
- TP/FN: 185 / 10
- TN/FP: 167 / 28

### Best
- M055: **0.6534**
- M068 HGB: **0.6568**
- Gain: **+0.0034**

**Conclusion:** NOT DEPLOYED.

The OOF gain was too small to justify replacing the already hidden-validated M067 deployment.

---

## M069 — M055 hidden-threshold sweep

**Goal:** Test whether the M055 decision threshold could be improved directly against hidden validation.

Known baseline:
- Threshold: **0.4989**
- Hidden score: **0.6322466368**

### Threshold 0.45
- Hidden score: **0.6301413737**
- Delta vs baseline: **-0.0021053**

### Threshold 0.55
- Hidden score: **0.6267967217**
- Delta vs baseline: **-0.0054499**

**Conclusion:** DISCARDED.

Both directions reduced hidden performance. Restored threshold **0.4989**.

The deployed threshold was already close to the hidden optimum, so further micro-threshold tuning was stopped.

---

## M070 — Challenge-specific NLI fine-tuning

**Goal:** Move beyond generic NLI and fine-tune DeBERTa-v3-base directly on competition-specific evidence/non-evidence relationships.

**Base model**
- `cross-encoder/nli-deberta-v3-base`

**Training strategy**
- Preserve pretrained 3-class NLI head.
- Convert logits into a binary entailment-vs-non-entailment signal:

`binary_logit = entailment_logit - logsumexp(contradiction_logit, neutral_logit)`

Training pairs included:
- Positive question + overlapping evidence candidate → positive.
- Positive question + high-ranked wrong candidate → negative.
- Hard-negative question + plausible transcript candidate → negative.
- Off-topic question + plausible transcript candidate → negative.

Conversation-disjoint 5-fold CV was retained.

### M070 direct classifier
- Accuracy: **0.8513**
- Scored tIoU: **0.4959**
- Composite: **0.6380**

Direct use of M070 was worse than M055.

### M055 + M070 fusion
Fold weights on M055:
- 0.60
- 0.70
- 0.70
- 0.70
- 0.20

Fold thresholds:
- 0.3936
- 0.3053
- 0.3053
- 0.2958
- 0.1934

OOF result:
- Accuracy: **0.9077**
- Scored tIoU: **0.5049**
- Composite: **0.6660**

Reference:
- M055 composite: **0.6534**
- Gain: **+0.0126**

**Conclusion:** PROMISING / DEPLOYMENT CANDIDATE.

Unlike the evidence-only experiments, task-specific NLI produced a substantial OOF improvement and appeared complementary to M055.

---

## M071 — M055 + full-data M070 deployment

**Goal:** Deploy the successful M070 fusion and test whether challenge-specific fine-tuning transfers to hidden validation.

### Full-data M070 export
- Training questions: **390**
- Training pairs: **2,788**
- Base model: `cross-encoder/nli-deberta-v3-base`
- Saved model: `medical/artifacts/models/m070_task_specific_nli`
- Entailment index: 1
- Contradiction index: 0
- Neutral index: 2

Deployment fusion:
- M055 weight: **0.70**
- M070 weight: **0.30**
- Threshold: **0.3053**

Evidence selector remained unchanged from M067.

### Smoke test
- HTTP: 200
- Structure: PASS
- Accuracy: 9/10
- Latency: **12.51 s**
- Well below 60 s budget.

### Hidden validation
- M049: **0.6178391635**
- M067: **0.6322466368**
- M071: **0.6727151494**

Improvements:
- vs M067: **+0.0404685**
- vs original M049: **+0.0548760**

**Conclusion:** ACCEPTED — CURRENT CHAMPION.

This was by far the largest hidden-validation improvement. The competition-specific fine-tuning generalized substantially better than expected from the OOF gain alone.

Protected deployment backups were created for M071.

---

## M072 — Self hard-negative mining

**Goal:** Improve M070 by using a first-stage task-specific model to mine its own hardest incorrect candidates from a broader candidate pool, then retrain a fresh stage-2 model.

**Leakage control**
For every outer CV fold:
1. Train stage-1 M070 only on outer-training conversations.
2. Score a broader top-40 candidate pool from those training conversations.
3. Mine high-scoring wrong spans.
4. Train a fresh stage-2 DeBERTa model on original + mined pairs.
5. Evaluate only on the held-out conversations.

Runtime evaluation remained top-12 candidates.

### Mining summary

| Fold | Original pairs | Mined pairs | Total pairs | Mean mined score |
|---:|---:|---:|---:|---:|
| 1 | 2190 | 2158 | 3932 | 0.0497 |
| 2 | 2221 | 2168 | 3980 | 0.0569 |
| 3 | 2224 | 2170 | 3977 | 0.0684 |
| 4 | 2244 | 2180 | 3985 | 0.0568 |
| 5 | 2273 | 2244 | 4084 | 0.0450 |

Fusion weights:
- 0.00
- 0.60
- 0.60
- 0.70
- 0.00

Fusion thresholds:
- 0.0008
- 0.3877
- 0.4123
- 0.2822
- 0.0008

Results:
- Gain vs M055: **+0.0106**
- Gain vs M070 fusion: **-0.0020**
- Approx. M072 fusion composite: **0.6640**
- Wall time: **2827.6 s (~47 min)**

**Conclusion:** DISCARDED.

The mined examples were mostly not genuinely difficult:
- mean mined M070 score only ~0.045–0.068.

The large number of additional easy negatives diluted the useful supervision. Fold fusion behavior was also unstable.

---

## M073 — Same-conversation contrastive negative strategy

**Goal:** Replace weak self-mined negatives with much stronger label-grounded semantic confounders from the same medical conversation.

### Contrastive-pair audit

Positive evidence spans:
- **195**

Total same-conversation question/evidence pairs:
- **1,950**

Targets:
- Positive: **195**
- Negative: **1,755**

Negative pairs by question type:
- Positive questions paired with another positive's evidence: **860**
- Hard-negative questions: **638**
- Off-topic questions: **257**

Pairs per transcript:
- Mean: 50
- Median: 50
- Min: 30
- Max: 70

Examples of potentially valuable contradictions:

- Q: `Were abnormal sounds heard over the lungs?`
- E: `Your lungs and your heart both sound normal. Nothing abnormal on listening.`

- Q: `Has the doctor found the condition to be unstable?`
- E: `No, nothing new.`

- Q: `Is the stomach acid medication being discontinued?`
- E: real treatment/evidence passages from the same consultation.

These are substantially stronger negatives than M072's low-confidence mined spans because they often express the same medical concept with the wrong polarity, value, treatment, or finding.

**Current status**
- Audit complete.
- Full M073 CV NOT started because competition time remaining is limited.
- Planned next step: fast one-fold pilot using only the most semantically similar same-conversation negatives.
- M071 (`0.6727151494` hidden) remains untouched as the protected champion.

---

### M073 fast contrastive pilot

Fold 1:
- M071 baseline: acc 0.9250, tIoU 0.4209, composite 0.6225
- M073: acc 0.9000, tIoU 0.4180, composite 0.6108
- Delta: -0.0118
- Predictions changed: 2
- Fixes: 0
- Breaks: 2
- Runtime: 185.1 s

Conclusion: DISCARDED. Same-conversation contrastive negatives degraded the held-out fold; no second fold or full training performed.

---

### M074 — Global M055/M070 fusion optimization

Global OOF search found a slightly better fixed deployment rule than M071:

- Previous M071:
  - M055 weight: 0.70
  - M070 weight: 0.30
  - threshold: 0.3053
  - OOF composite: 0.673696
  - hidden validation: 0.6727151494

- M074:
  - M055 weight: 0.68
  - M070 weight: 0.32
  - threshold: 0.3226589362
  - OOF composite: 0.674722
  - hidden validation: 0.6748204126

Hidden gain vs M071: +0.0021053.

Conclusion: ACCEPTED. M074 is the new hidden-validation champion.

---

### M075 — Learned M055/M070 meta-fusion

Tested HGB and logistic meta-models using M055 probability plus M070 score-distribution features.

Reference M074:
- Accuracy: 0.9205
- tIoU: 0.5109
- Composite: 0.6747

HGB:
- Accuracy: 0.9256
- tIoU: 0.4976
- Composite: 0.6688

Logistic:
- Accuracy: 0.9205
- tIoU: 0.5110
- Composite: 0.6748

Best gain vs M074: +0.0001.

Conclusion: DISCARDED. The simple M074 linear fusion already captures essentially all useful fusion signal.

### M076 — Multi-passage question classifier pilot

Architecture:
- DeBERTa-v3-base
- Fresh binary classification head
- Input: question + top 5 M047 passages
- Direct task: predict YES/NO for the competition question

Fold 1:
- M074 baseline: acc 0.9250, tIoU 0.4209, composite 0.6225
- M076 direct: acc 0.9000, tIoU 0.4209, composite 0.6125
- M074/M076 fusion: acc 0.9500, tIoU 0.4209, composite 0.6325
- Delta vs M074: +0.0100
- Best M076 weight: 0.30
- Runtime: 159 s

Status: PROMISING BUT UNCONFIRMED. Fold 2 required before deployment.

---

### M077 — M074 + M076 multi-passage fusion deployment

M076 was trained on the full 390-question dataset using the question plus the top 5 retrieved passages as one direct YES/NO classification input.

Deployment fusion:
- M074 weight: 0.65
- M076 weight: 0.35
- Fusion threshold: 0.3847283086
- Evidence selector unchanged.

Smoke test:
- HTTP 200
- Structure PASS
- Accuracy: 10/10
- Latency: 14.03 s

Notably, M076 corrected the previous sample false positive:
- "Were abnormal sounds heard over the lungs?"
- M074: True
- M077: False

Hidden validation:
- M074: 0.6748204126
- M077: 0.6997446225
- Gain: +0.0249242099

Conclusion: ACCEPTED. M077 is the new hidden-validation champion.

---

### M078 — Full 5-fold M076 OOF calibration

Current M077 OOF composite: 0.6810

Nested leave-one-fold-out tuning:
- Composite: 0.6822
- Gain: +0.0011

Fold M076 weights:
- 0.37
- 0.37
- 0.42
- 0.35
- 0.37

Fold thresholds:
- 0.450365
- 0.450365
- 0.569985
- 0.445497
- 0.450365

Median fixed deployment:
- M076 weight: 0.37
- threshold: 0.450364832
- OOF composite: 0.6892
- OOF gain vs M077: +0.0082

Hidden validation:
- M077: 0.6997446225
- M078: 0.6963999705

Conclusion: DISCARDED. The stronger OOF calibration did not transfer to hidden validation.

### M079 — 3-epoch M076 pilot

Tested whether extending M076 fine-tuning from 2 to 3 epochs improved fold 1.

Fold 1:
- M074 baseline: 0.6225
- 2-epoch fusion: 0.6325
- 3-epoch fusion: 0.6325

Delta vs 2 epochs: +0.0000.

Conclusion: DISCARDED. No evidence that a third epoch improves generalization.