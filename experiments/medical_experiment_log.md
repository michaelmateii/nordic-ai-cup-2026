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
