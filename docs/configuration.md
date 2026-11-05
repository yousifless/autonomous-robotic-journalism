# Configuration and tuning

Every setting is a ROS 2 parameter in [`config/params.yaml`](../src/journalist_robot/config/params.yaml). Values
reported in the paper are marked `[paper]` there. Change them in a copy of the file and pass it
with `params_file:=/path/my_params.yaml`. Only the launch arguments listed by
`ros2 launch journalist_robot journalist_system.launch.py -s` (mission, topics, rotation, output folder, which nodes
to start) can be set directly on the command line; an unknown `name:=value` is silently ignored by `ros2 launch`.
ROS 2 parameters are typed: write floating-point values with a decimal point (`10.0`, not `10`).

The paper presents its thresholds and weights as prototype settings that were not independently calibrated. Treat the
defaults as a starting point and check their effect in `decisions.jsonl` for your environment.

## Motion

| Parameter | Default | Effect |
|---|---|---|
| `max_linear`, `max_angular` | 0.4 m/s, 0.5 rad/s | hard limits after the policy; the prompt asks the model for speeds within them |
| `decision_period_s`, `decision_window_s` | 2.0 s | how often and over how much video the controller decides |
| `frames_per_decision` | 4 | frames given to VideoLLaMA3 per decision; fewer frames lower latency |
| `turn_cooldown_s` | 1.0 s | minimum time before reversing turn direction |
| `deadman_timeout_s` | 5.0 s | zero velocity if no decision arrives for this long; keep it above the worst inference time |
| `max_frame_age_s` | 3.0 s | zero velocity (and no decision) if the newest camera frame is older than this |
| `allow_reverse` | false | allow negative `linear_x`; off by default because decisions see only the front camera |

## Newsworthiness (Eq. 1) and the recording trigger

| Parameter | Default | Effect |
|---|---|---|
| `newsworthiness_weights` | [0.4, 0.4, 0.2] | w1 intrinsic, w2 brief, w3 novelty; must sum to 1 |
| `brief_similarity_range` | [0.15, 0.35] | raw CLIP image-text similarity mapped to [0, 1] for S_brief |
| `novelty_buffer_size` | 8 | number of past views S_novelty compares against |
| `start_newsworthiness`, `start_confidence` | 0.7, 0.8 | recording starts when both are reached |
| `stop_newsworthiness`, `stop_confidence` | 0.3, 0.5 | recording stops when both fall below |
| `max_segment_duration_s` | 0 (off) | optional hard limit on segment length |

Expected effects of the weights, with the thresholds fixed:

- raising **w2** concentrates recording on views that match the brief (better assignment adherence, but unexpected
  events may be missed);
- raising **w1** admits generically salient activity even when it is off-brief;
- raising **w3** favours views not seen recently (broader exploration, but less sustained coverage of one subject).

Since articles are written only from recorded segments, these changes also change what the article can contain.
These are expected effects of the formulation; the paper does not report a weight-sensitivity study.

If recording never starts, inspect `newsworthiness`, `intrinsic`, `brief_relevance` and `confidence` in
`decisions.jsonl`: a brief whose first paragraph does not describe the visual targets gives low S_brief.

## Frame selection

| Parameter | Default | Effect |
|---|---|---|
| `similarity_threshold` | 0.87 | a frame is kept only if its CLIP similarity to the last kept frame is below this |
| `min_selection_interval_s` | 0.4 s | minimum spacing between kept frames |
| `batch_size` | 8 | frames per description query |
| `selection_enabled` | true | false keeps every frame (slower description; useful as a baseline) |

Selection reduces description time; it does not guarantee that discarded frames carried no relevant evidence. To
measure that trade-off, run the same footage with `selection_enabled` true and false.

## Acceptance gate (Eq. 4)

| Parameter | Default | Effect |
|---|---|---|
| `acceptance_weights` | [0.35, 0.25, 0.20, 0.20] | factuality, relevance, consistency, language; must sum to 1 |
| `acceptance_threshold` | 0.85 | drafts at or above are accepted |
| `max_rewrites` | 2 | rewrites before a draft is left, marked not accepted, for a human editor |
| `max_key_frames` | 16 | frames the visual signals use |
| `evaluation_enabled` | true | false skips evaluation (draft saved as "not evaluated") |

A larger factuality weight makes the gate stricter toward weakly grounded drafts (more rewriting); a larger language
weight can let fluent but weakly grounded drafts through. The score is internal and model-based. Every final draft is saved and
shown in the UI with its score and an accepted / not accepted label; acceptance means the draft passed the internal
gate, not that every claim is true.

## Article

`article_min_words` / `article_max_words` (800 / 1000) set the requested length; `writer_max_new_tokens` (8192) must
leave room for DeepSeek-R1's reasoning trace plus the article. If the trace does not finish within that budget, the
writer retries once and otherwise saves the raw output under `drafts/` without publishing an article.
`writer_max_evidence_chars` (60000) bounds the evidence given to the writer: for longer segments the batch
descriptions are thinned out evenly over the segment. `max_pending_batches` (32) bounds the description backlog; if
the describer falls that far behind, further batches are not described (the frames are still saved) and a warning
suggests a stricter selection.

## Audio

Set `audio_enabled: true` and point `audio_topic` at any node publishing `audio_common_msgs/AudioData` with
little-endian 16-bit PCM at `audio_sample_rate` and `audio_channels`. Robot and server clocks must be synchronised.

## Output and UI

`output_root`, `save_decision_frames`, `speech_output_enabled` (article.wav via pyttsx3), `ui_enabled`, `ui_host`,
`ui_port`. Keep `ui_host` at `127.0.0.1` unless the network is trusted; use an SSH tunnel for remote viewing:

```bash
ssh -L 8890:127.0.0.1:8890 user@gpu-server   # then open http://127.0.0.1:8890/
```
