# Captured-view check returned by the journalism pipeline to the controller.
#
# When the captured view is unclear or off-topic, the journalist publishes
# actionable guidance (e.g. "rotate left toward the presenter") that the
# controller injects into its next prompts.

std_msgs/Header header

uint32 segment_id

bool view_usable            # subject is clearly visible
bool on_topic               # view matches the mission brief

# One of: NONE | APPROACH | HOLD | ROTATE_LEFT | ROTATE_RIGHT | BACK_OFF
string suggested_action

# Free-text guidance for the controller prompt.
string guidance
