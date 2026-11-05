# One decision of the perception & action loop (Sec. III-A of the paper).
#
# Published by perception_controller_node on every control cycle, after the
# safety/control policy has been applied to the VLM proposal.

std_msgs/Header header

# Velocity command actually sent to the base (after clamping and gating).
float32 linear_x            # m/s, positive = forward
float32 angular_z           # rad/s, positive = turn left

# Controller self-assessment returned by the VLM, in [0, 1].
float32 confidence

# Newsworthiness N_t (Eq. 1) and its three components, each in [0, 1].
float32 newsworthiness
float32 intrinsic_score     # S_news   : VLM-rated journalistic value
float32 brief_relevance     # S_brief  : CLIP similarity to the mission brief
float32 novelty             # S_novelty: 1 - max CLIP similarity to recent views

# Recording trigger produced by the hysteresis rule.
string TRIGGER_START=START
string TRIGGER_CONTINUE=CONTINUE
string TRIGGER_STOP=STOP
string trigger

# Discretised movement label: FORWARD | BACKWARD | TURN_LEFT | TURN_RIGHT | HOLD
string movement_label

# Natural-language reasoning returned by the VLM (plus policy annotations).
string reasoning

# Wall-clock time spent on model inference for this decision.
float32 inference_latency_s
