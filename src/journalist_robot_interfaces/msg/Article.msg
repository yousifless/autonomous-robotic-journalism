# A generated article draft and its internal acceptance result (Sec. III-C).
#
# The acceptance score is an internal heuristic computed by models that are
# partly reused from the perception pipeline. It is not an independent
# assessment of factual accuracy.

std_msgs/Header header

uint32 segment_id

string headline
string body

# Internal acceptance score S (Eq. 4) and its four signals, each in [0, 1].
float32 acceptance_score
float32 factuality
float32 relevance
float32 consistency
float32 language

bool accepted               # S >= acceptance threshold
uint32 rewrite_count        # number of rewrites performed before this draft

# Seconds between the end of data collection and the final draft.
float32 generation_time_s

# Directory holding the article, evidence and evaluation report on disk.
string output_dir
