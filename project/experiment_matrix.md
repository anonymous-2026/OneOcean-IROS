# Experiment Matrix

Use this file as the source of truth for **what you will run** and **why**.
The goal is to map *claims* to *falsifiable tests* with clear metrics and baselines.

## Claims (2–5)

1. **Claim C1 (testable)**: TODO
2. **Claim C2 (testable)**: TODO

## Matrix

| Experiment ID | Claim(s) | Purpose | Dataset/Benchmark | Baselines | Metrics | Seeds | Expected outcome | Run IDs (fill after) |
|---|---|---|---|---|---|---:|---|---|
| E1 | C1 | Main comparison | TODO | TODO | TODO | 3–5 | TODO | TODO |
| E2 | C1 | Ablation | TODO | TODO | TODO | 3–5 | TODO | TODO |
| E3 | C2 | Robustness / generalization | TODO | TODO | TODO | 3–5 | TODO | TODO |

## Reporting notes

- Prefer 3–5 seeds when feasible; if single-seed, label as exploratory.
- Save metrics in `runs/<run_id>/metrics.json` (or a small `eval.csv`) for aggregation.
- Always record failure cases and training instability notes in `runs/<run_id>/notes.md`.

