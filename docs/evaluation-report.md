# Offline evaluation report

Run from the repository root (no extra Python dependencies):

```bash
python scripts/build_eval_report.py --run-dir outputs/evaluation/RUN_NAME
```

Open `outputs/evaluation/RUN_NAME/report.html` in a recent Chrome, Edge,
Firefox or Safari. The HTML includes compressed data and needs no network.
It supports outcome/persona/difficulty/domain filters, charts, search,
per-task conversations and tool traces, pagination, and filtered CSV export.

Optional arguments:

- `--output /path/to/report.html`: custom output location.
- `--title "Experiment title"`: report title.
- `--expected-tasks N`: explicitly known intended run scope; omit if unknown.
- `--tasks-file outputs/split/tasks_final.jsonl`: task metadata source.

Metrics are recomputed from trajectories using the project's strict Reward v3
success definition. The last record in file order wins for duplicate task IDs.
Infrastructure errors remain in the observed task denominator. Without an
explicit expected task count, rates describe observed tasks only. Missing
metadata is shown as unknown. A stale summary is flagged, never overwritten.
An unfinished final JSONL line is ignored; malformed internal records fail.

This is a snapshot: rerun after evaluation advances. Source inputs are never
modified, and generating a report does not start evaluation or training.
Filters update charts/table; headline totals always describe the whole snapshot.
The HTML contains full evaluation conversations; consider this when sharing it.

Validation:

```bash
python -m unittest discover -s tests -p test_eval_report.py
```
