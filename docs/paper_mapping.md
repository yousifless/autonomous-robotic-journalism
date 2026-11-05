# Paper-to-code map

Where each element of the SII 2027 paper is implemented in this release.

| Paper | Description | Code |
|---|---|---|
| Sec. III, Fig. 2 | two-loop architecture | `nodes/` (three nodes), [architecture.md](architecture.md) |
| Sec. III-A, prompt box | mission-conditioned controller prompt | `core/prompts.py` (`CONTROLLER_SYSTEM`, `controller_prompt`) |
| Sec. III-A | ~2 s windows to VideoLLaMA3, structured output | `models/videollama3_controller.py`; `decision_window_s`, `decision_period_s` |
| Sec. III-A | summary of recent commands | `core/control_policy.py` (`ExplorationState`) |
| Eq. (1) | N_t = w1 S_news + w2 S_brief + w3 S_novelty | `core/newsworthiness.py`; CLIP in `models/clip_encoder.py` |
| Eq. (2) | S_brief = CLIP image-text similarity | `NewsworthinessScorer.brief_relevance` |
| Eq. (3) | S_novelty = 1 − max similarity to buffer B_t | `NewsworthinessScorer.novelty` |
| Weight selection | w = 0.4 / 0.4 / 0.2 | `newsworthiness_weights` |
| Control policy | start N_t ≥ 0.7 & c ≥ 0.8, stop N_t < 0.3 & c < 0.5 | `core/control_policy.py` (`RecordingTrigger`) |
| Control policy | forward when clear and newsworthy, otherwise rotate | `ControlPolicy`, prompt decision rules |
| Sec. III-B | 10 s pre-event window, dual camera | `nodes/capture_manager_node.py` |
| Sec. III-B | captured-view quality / relevance check → feedback | `journalism/view_check.py`, `JournalistNode._check_view`, `CaptureFeedback` |
| Sec. III-B, Table I | CLIP ViT-B/32 semantic selection, 87 % threshold | `journalism/frame_selector.py` |
| Sec. III-B | batches of 8, LLaVA-NeXT-Interleave, previous batch as context | `models/frame_describer.py`, `JournalistNode._describe_batch` |
| Sec. III-B | Whisper, shared timeline | `models/transcriber.py`, `JournalistNode._transcribe_until` |
| Sec. III-B | asynchronous description during collection | selection and description threads in `journalist_node.py` |
| Sec. III-B | retrieval-augmented grounding, DeepSeek-R1, inverted pyramid | `core/prompts.py` (`ARTICLE_PROMPT`), `models/article_writer.py`, `ui/rag_index.py` |
| Sec. III-B | text and audio output | `article.md`, `models/speech_output.py` |
| Sec. III-C, Table II | evaluation dimensions and models | `evaluation/signals.py` |
| Eq. (4) | S = 0.35 S_fact + 0.25 S_rel + 0.20 S_cons + 0.20 S_lang, accept S ≥ 0.85 | `evaluation/acceptance.py`, `evaluation/evaluator.py` |
| Sec. III-C | revision when S < 0.85 | rewrite loop in `JournalistNode._finish_segment` (`max_rewrites`) |
| Sec. III-C | chat interface with RAG | `ui/chat_server.py` |
| Sec. IV-A | Jetson AGX Orin, remote A100, ROS 2, velocity commands | `launch/`, [installation.md](installation.md) |
| Sec. IV-B1 | 70-decision budget, Left/Right/Forward/Stop discretisation, alignment | `analysis/decision_metrics.py` |
| Sec. IV-B2, Table IV | behavioural divergence between two briefs | `decision_metrics` on two runs |
| Sec. V-C | post-collection article time | `generation_time_s` in `article.json` and `Article` |

## Implementation choices not specified in the paper

These are settings of this release. Most are parameters in `params.yaml`; the S_fact and S_lang composition and the
S_rel rescaling are constructor arguments in `evaluation/signals.py`, and the command labelling is the `--precedence`
option of `decision_metrics`.

| Choice | Value |
|---|---|
| Frames per controller decision | 4 frames sampled from the 2 s window |
| Rescaling of raw CLIP similarity for S_brief and S_rel | linear from [0.15, 0.35] to [0, 1] |
| Text used as the CLIP query for S_brief | first paragraph of the brief (CLIP reads at most 77 tokens) |
| Novelty buffer size | 8 views |
| Composition of S_fact | 0.6 visual grounding + 0.4 source adherence |
| Composition of S_lang | mean of coherence, grammar and structure scores |
| Maximum number of rewrites | 2, after which the draft is marked as requiring human editing |
| Deadman timeout and command re-publish rate | 5 s, 10 Hz |
| Labelling of commands that both rotate and translate | rotation takes precedence (`--precedence translation` to change) |
