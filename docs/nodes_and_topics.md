# Nodes, topics and messages

## Topic graph

| Topic | Type | Publisher | Subscribers | QoS |
|---|---|---|---|---|
| `/front_camera/image_raw/compressed` | `sensor_msgs/CompressedImage` (or `Image` with `image_transport:=raw`) | camera driver | `perception_controller`, `capture_manager` | sensor data |
| `/rear_camera/image_raw/compressed` | same | camera driver | `capture_manager` | sensor data |
| `/audio/audio` | `audio_common_msgs/AudioData` (16-bit PCM) | microphone driver | `journalist` (if `audio_enabled`) | depth 100 |
| `/cmd_vel` | `geometry_msgs/Twist` | `perception_controller` | base driver | depth 10 |
| `/journalist/decision` | `journalist_robot_interfaces/ControlDecision` | `perception_controller` | `capture_manager` | reliable, depth 50 |
| `/journalist/segment_events` | `journalist_robot_interfaces/SegmentEvent` | `capture_manager` | `journalist` | reliable, depth 50 |
| `/journalist/segment_frames` | `sensor_msgs/CompressedImage` (JPEG, `frame_id = segment_<id>`) | `capture_manager` | `journalist` | reliable, depth 500 |
| `/journalist/capture_feedback` | `journalist_robot_interfaces/CaptureFeedback` | `journalist` | `perception_controller` | reliable, depth 50 |
| `/journalist/article` | `journalist_robot_interfaces/Article` | `journalist` | your application | reliable, depth 50 |
| `/journalist/autonomy_enable` | `std_msgs/Bool` | you | `perception_controller` | reliable, depth 50 |

Camera and `cmd_vel` topic names are parameters (and launch arguments); the `/journalist/*` names are fixed and can
be remapped with standard ROS 2 remapping.

## Messages (`journalist_robot_interfaces`)

**`ControlDecision`** — one control cycle: `linear_x`, `angular_z` (sent command), `confidence`, `newsworthiness`
(N_t) with `intrinsic_score`, `brief_relevance`, `novelty`, `trigger` (`START` / `CONTINUE` / `STOP`),
`movement_label`, `reasoning`, `inference_latency_s`.

**`SegmentEvent`** — `type` (`START=0`, `END=1`, `SHUTDOWN=2`), `segment_id`, `pre_event_frames`, `mission_brief`.

**`CaptureFeedback`** — `segment_id`, `view_usable`, `on_topic`, `suggested_action`
(`NONE | APPROACH | HOLD | ROTATE_LEFT | ROTATE_RIGHT | BACK_OFF`), `guidance`.

**`Article`** — `segment_id`, `headline`, `body` (lead + body), `acceptance_score` with `factuality`, `relevance`,
`consistency`, `language`, `accepted`, `rewrite_count`, `generation_time_s` (end of collection to final draft),
`output_dir`.

## Executables

| Executable | Description |
|---|---|
| `perception_controller` | VideoLLaMA3 controller node |
| `capture_manager` | dual-camera composition, pre-event buffer, segments |
| `journalist` | journalism pipeline, evaluation, web UI |
| `decision_metrics` | offline alignment / divergence between two `decisions.jsonl` files |

## Parameters

All parameters, with defaults and the values reported in the paper, are listed and commented in
[`config/params.yaml`](../src/journalist_robot/config/params.yaml). The launch file additionally sets, for all nodes:

| Launch argument | Node parameter | Default |
|---|---|---|
| `mission` | `mission_brief_file` (controller, capture manager) | `research_lab_people_working.txt` |
| `output_root` | `output_root` | `~/journalist_robot_runs` |
| `run_id` | `run_id` (same for all nodes) | launch date and time |
| `front_image_topic`, `rear_image_topic`, `image_transport`, `rotate_180` | same | see params.yaml |
| `cmd_vel_topic` | `cmd_vel_topic` | `/cmd_vel` |
| `start_controller`, `start_capture`, `start_journalist` | – | `true` |
| `params_file` | – | installed `params.yaml` |

## Running nodes on different machines

```bash
# GPU 1 / machine A
ros2 launch journalist_robot journalist_system.launch.py start_journalist:=false mission:=...
# GPU 2 / machine B (same run_id so both write to the same run folder name)
ros2 launch journalist_robot journalist_system.launch.py start_controller:=false start_capture:=false run_id:=<same id>
```
