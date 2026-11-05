# Installation

## 1. Hardware

| Component | Used in the paper's prototype | Requirement |
|---|---|---|
| Mobile base | differential-drive base driven by velocity commands | any base whose ROS 2 driver subscribes to `geometry_msgs/Twist` on `/cmd_vel` |
| Cameras | front- and rear-facing cameras on a sensor mast at approximately eye height | one front camera (required) and one rear camera (optional) |
| Microphone | microphone array | optional; any node publishing raw 16-bit PCM as `audio_common_msgs/AudioData` |
| On-board computer | NVIDIA Jetson AGX Orin | runs camera and base drivers |
| GPU server | NVIDIA A100 | all models; see memory budget below |
| Network | campus Wi-Fi | both machines on the same ROS 2 network |

### GPU memory budget (approximate)

| Model | Precision | Approx. memory |
|---|---|---|
| VideoLLaMA3-7B (controller) | bf16 | ~17 GB |
| DeepSeek-R1-Distill-Qwen-32B (writer) | 4-bit NF4 | ~20 GB |
| LLaVA-NeXT-Interleave-Qwen-7B (describer, checks) | 4-bit | ~6 GB |
| CLIP ViT-B/32 (x2 processes), BLIP-VQA, RoBERTa-large-MNLI, Sentence-BERT, Whisper-small | fp32/fp16 | ~5 GB |

A single 80 GB GPU holds everything. With less memory, run the `journalist` node on a second GPU or machine
(`start_journalist:=false` on one machine, `start_controller:=false start_capture:=false` on the other), or set
`evaluation_enabled: false` (in your copy of `config/params.yaml`) to skip the evaluation models.

## 2. ROS 2

Install **ROS 2 Humble** on Ubuntu 22.04 on both machines following the
[official instructions](https://docs.ros.org/en/humble/Installation.html), then:

```bash
sudo apt install python3-colcon-common-extensions python3-rosdep python3-opencv
# Robot side, for the example camera launch file:
sudo apt install ros-humble-v4l2-camera ros-humble-image-transport-plugins
```

Only if you use a microphone (`audio_enabled: true`): build or install the `audio_common_msgs` package (from the
ROS 2 branch of [ros-drivers/audio_common](https://github.com/ros-drivers/audio_common), or your distribution's
binary package if one is available) on both machines. The journalist expects raw 16-bit PCM, so start the audio
capture node with `format:=wave` and `sample_format:=S16LE` (its default is mp3), with the sample rate and channel
count set in `audio_sample_rate` / `audio_channels`.

## 3. Python dependencies (GPU server)

```bash
python3 -m pip install -r requirements.txt
# FlashAttention 2 (optional but recommended; needs a matching CUDA toolkit):
python3 -m pip install flash-attn --no-build-isolation
```

Without FlashAttention, set `use_flash_attention: false` for the `perception_controller` and `journalist` nodes in your copy of `config/params.yaml`.

LanguageTool (grammar signal) needs Java. Without it, the grammar part of the language signal falls back to a
simple heuristic (recorded as `"heuristic"` in the evaluation report). You can also point `language_tool_url` at a
LanguageTool server.

## 4. Models

All models are downloaded from the Hugging Face Hub on first use:

| Parameter | Default |
|---|---|
| `controller_model_id` | `DAMO-NLP-SG/VideoLLaMA3-7B` |
| `clip_model_id` | `openai/clip-vit-base-patch32` |
| `describer_model_id` | `llava-hf/llava-interleave-qwen-7b-hf` |
| `writer_model_id` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` |
| `asr_model_id` | `openai/whisper-small` |
| evaluation (fixed) | `Salesforce/blip-vqa-base`, `roberta-large-mnli`, `sentence-transformers/all-mpnet-base-v2`, `sentence-transformers/all-MiniLM-L6-v2` |

To pre-download them (recommended before going on the robot), run once with internet access, e.g.:

```bash
for m in DAMO-NLP-SG/VideoLLaMA3-7B deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
         llava-hf/llava-interleave-qwen-7b-hf openai/clip-vit-base-patch32 openai/whisper-small \
         Salesforce/blip-vqa-base roberta-large-mnli \
         sentence-transformers/all-mpnet-base-v2 sentence-transformers/all-MiniLM-L6-v2; do
  huggingface-cli download "$m"
done
```

Afterwards set `local_files_only: true` (for both `perception_controller` and `journalist`) in your copy of
`config/params.yaml` to run fully offline; this also switches the Hugging Face hub to offline mode. LanguageTool
downloads its server on first use, so run the evaluation once online or point `language_tool_url` at a local
LanguageTool server; without it the grammar signal falls back to a simple heuristic. VideoLLaMA3 loads custom code
from its model repository (`trust_remote_code=True`); review it before use.

## 5. Build

```bash
mkdir -p ~/journalist_ws/src && cd ~/journalist_ws/src
git clone <this-repository-url> autonomous_robotic_journalism
cd ~/journalist_ws
rosdep install --from-paths src --ignore-src -y
colcon build --symlink-install
source install/setup.bash
```

Build the workspace on both machines (the robot only needs `journalist_robot_interfaces` and the launch file;
`colcon build --packages-select journalist_robot_interfaces journalist_robot` is enough).

## 6. Networking

- Use the same `ROS_DOMAIN_ID` on both machines and make sure DDS discovery works over your Wi-Fi
  (multicast is often blocked on campus networks; use a Fast DDS discovery server or a Cyclone DDS peer list).
- Publish camera images **compressed** over Wi-Fi (`image_transport:=compressed`, the default). Raw images saturate
  most wireless links.
- **Synchronise clocks** (e.g. `chrony`) between the robot and the GPU server. Frames carry the camera's timestamps
  and audio is stamped on the GPU server, so speech only lines up with the frames if the clocks agree.

## 7. Check the installation

```bash
cd ~/journalist_ws/src/autonomous_robotic_journalism/src/journalist_robot
python3 -m pytest test                                   # model-independent unit tests
ros2 interface show journalist_robot_interfaces/msg/ControlDecision
ros2 launch journalist_robot journalist_system.launch.py --show-args
```

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `No camera frames yet` warnings | wrong `front_image_topic`, wrong `image_transport`, or DDS discovery failing across machines |
| Robot does not move | `autonomy_enabled` false, base driver not on `/cmd_vel`, or confidence below the gating thresholds (see `decisions.jsonl`) |
| Robot stops every few seconds | inference slower than `deadman_timeout_s`; increase it or reduce `frames_per_decision` |
| Recording never starts | N_t or confidence never reach 0.7 / 0.8; inspect the `newsworthiness`, `brief_relevance` and `confidence` fields in `decisions.jsonl`, check the mission brief, or adjust `brief_similarity_range` |
| Images upside down | set `rotate_180:=true` |
| CUDA out of memory | split nodes across GPUs, keep the 4-bit options on, or disable evaluation |
| `audio_common_msgs is not installed` | install `audio_common_msgs` (Section 2) or keep `audio_enabled: false` |
