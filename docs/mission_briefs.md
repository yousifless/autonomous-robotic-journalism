# Mission briefs

A mission brief is a plain-text file that tells the robot what the editor wants covered. It is:

- injected verbatim into every controller prompt (subjects, rubric, priorities, safety);
- used, through its **first paragraph**, as the CLIP text query for S_brief in Eq. (1);
- passed to the describer, the captured-view check and the article writer.

The file is re-read when it changes, so the assignment can be updated during a run.

## Bundled briefs (`src/journalist_robot/missions/`)

| File | Assignment |
|---|---|
| `research_lab_people_working.txt` | document people working at desks in a research lab |
| `robotics_workshop_research_advancements.txt` | document robots, experiments, prototypes and instrumentation in a robotics workshop |
| `office_furniture_and_workspaces.txt` | document furniture arrangements and workspace layouts |
| `TEMPLATE.txt` | skeleton for writing your own |

Use one with `mission:=<file name>` or pass any path: `mission:=/home/me/briefs/press_event.txt`.

## Writing a brief

Follow `TEMPLATE.txt`:

1. **Opening paragraph — make it visual.** State the place and the targets in concrete visual terms
   ("people sitting at desks working on computers"), not abstract goals ("productivity"). This paragraph is what CLIP
   compares with the camera image, and CLIP only reads about the first 77 tokens.
2. **Newsworthiness rubric** with HIGH (0.7–1.0), MEDIUM (0.4–0.7) and LOW (0.0–0.4) examples. Recording starts at
   N_t ≥ 0.7, so HIGH should describe exactly the views you want recorded.
3. **Priority targets**, in order.
4. **Exploration strategy and movement behaviour**: when to approach, when to scan.
5. **What to ignore.**
6. **Safety considerations** specific to the place (distance to people, areas to avoid).

Keep briefs factual and specific. A brief that asks for something the camera cannot see (e.g. "find the most
important person") leads to arbitrary ratings.

## Two briefs, one environment

The paper's context-variation experiment ran the same environment under two briefs (e.g. *people working* vs.
*empty desks* in the research lab) and compared the command sequences. To repeat it, run twice with different
`mission:=` values and compare the two `decisions.jsonl` files with `decision_metrics` (see [outputs.md](outputs.md)).
Different trajectories lead to different later observations, so this does not isolate the brief under identical
sensory input.
