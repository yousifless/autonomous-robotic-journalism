# Lifecycle event of a recording segment.
#
# Published by capture_manager_node. A segment begins with START (followed by
# the buffered pre-event frames), and ends with END. SHUTDOWN asks the
# journalist to finalise any open segment before the system stops.

std_msgs/Header header

uint8 START=0
uint8 END=1
uint8 SHUTDOWN=2
uint8 type

uint32 segment_id

# Number of pre-event frames that will be (START) or were (END) streamed for
# this segment. Informational only.
uint32 pre_event_frames

# Mission brief active when the segment started.
string mission_brief
