# Outputs

Every launch writes to `<output_root>/<run_id>/` (default `~/journalist_robot_runs/<date_time>/`).

```
<run_id>/
├── decisions.jsonl                 one JSON object per control decision
├── decision_frames/000000.jpg ...  newest input frame of each decision (save_decision_frames)
└── segment_001_142305/             segment id + start time (HHMMSS), unique within the run
    ├── frames/000000.jpg ...       frames kept by semantic selection (combined front | rear)
    ├── selection.json              frames received vs. kept, selection settings
    ├── descriptions.jsonl          one object per batch of 8 frames
    ├── transcript.txt              [start - end] speech lines (audio enabled)
    ├── feedback.jsonl              captured-view checks, and whether feedback was sent
    ├── drafts/
    │   ├── draft_0.md              first draft
    │   ├── draft_0_evaluation.json its four signals, S, and the problems found
    │   └── draft_1.md ...          rewrites
    ├── article.md                  final draft: headline, lead, body
    ├── article.json                final draft + scores + rewrite count + generation time + status
    └── article.wav                 spoken version (speech_output_enabled)
```

## `decisions.jsonl`

| Field | Meaning |
|---|---|
| `index`, `time` | decision number, wall-clock time |
| `linear_x`, `angular_z`, `label` | command sent to the base after the policy |
| `proposal` | the VLM's parsed answer before the policy (velocities, confidence, rating, reasoning, `parsed_with_fallback`) |
| `newsworthiness`, `intrinsic`, `brief_relevance`, `novelty` | N_t and its components (Eq. 1) |
| `confidence`, `trigger` | c_t and START / CONTINUE / STOP |
| `policy_notes` | what the policy changed and why |
| `inference_latency_s` | model time for this decision |
| `mission_brief_id` | short hash of the brief in force |
| `prompt`, `raw_model_output` | full model input text and output |

## `article.json`

`headline`, `lead`, `body`, `acceptance_score`, `factuality`, `relevance`, `consistency`, `language`, `accepted`,
`evaluated`, `rewrite_count`, `generation_time_s` (from the end of data collection to the final draft) and `status`
(`accepted`, `not evaluated`, or `below threshold after N rewrites - requires human editing`).

## Navigation metrics

`decision_metrics` reproduces the index-matched measures of the paper's pilot. Commands are discretised into Left
(ω > 0.2), Right (ω < −0.2), Forward (v > 0) or Stop and compared index by index over a decision budget.

```bash
# Alignment with a teleoperated reference run (Table III style)
ros2 run journalist_robot decision_metrics teleop/decisions.jsonl autonomous/decisions.jsonl --budget 70

# Divergence between two briefs in the same environment (Table IV style)
ros2 run journalist_robot decision_metrics brief_a/decisions.jsonl brief_b/decisions.jsonl --budget 70
```

`--precedence rotation|translation` sets which label wins when a command both rotates and translates (default:
rotation). For a teleoperated reference, record the operator's `/cmd_vel` at the same decision period and convert it
to the same JSONL fields (`linear_x`, `angular_z`), e.g. from a rosbag.

Matched decision indices need not correspond to matched scene states, and the 70 decisions of a run are not
independent; these metrics measure behavioural agreement, not navigation or reporting quality. Report them together
with an outcome measure such as subject visibility.
