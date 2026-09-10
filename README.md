# ARC Prize 2026 — ARC-AGI-3 Agent

Agent development for the [ARC Prize 2026 - ARC-AGI-3](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3)
Kaggle competition ($850K track): build an AI agent that plays 110 hidden,
instruction-free game environments — exploring, goal-finding, and winning levels
with human-competitive action efficiency.

## Current status

| Milestone | Result |
|---|---|
| Baseline `Explorer` agent (deterministic, CPU-only, zero LLM calls) | **0.69%** on the 25 public games |
| Kaggle submission pipeline | **validated end-to-end** (official starter kit, notebook pushed) |
| Game intelligence | all 25 public games statically profiled |
| Adaptive-budget sweep | done — see `notes/sweep_adaptive.md` |

Scoring (RHAE): per completed level `(human_baseline / agent_actions)^2` capped at 1.15,
weighted by level index; total = mean over games. Frontier AI ≈ 0.51%.

## Layout

```
ARC-AGI-3-Agents/        # official framework (forked state) + our additions:
  agents/templates/explorer.py   # <- the Explorer agent (memory graph, BFS replays,
                                 #    undo backtracking, give-up heuristics)
  main.py                        # + offline game-listing fallback
  tests/                         # fixed for arc_agi 0.9.8 APIs (59 passing)
ARC-AGI-3-Kaggle-Starter/  # official submission kit + our agent (agent/my_agent.py)
notes/                   # game profiles + sweep reports (JSON + Markdown)
sweeps/                  # adaptive-budget sweep runner
kaggle/                  # offline-tested notebook skeleton (superseded by the starter)
environment_files/       # 25 public game environments (from the Kaggle bundle)
```

Excluded by `.gitignore` (re-downloadable or reproducible): `arc_agi_3_wheels/`
(from the competition data page), virtualenvs, gameplay recordings, `vendor/`.

## Quickstart (local dev)

```bash
cd ARC-AGI-3-Agents
uv sync                              # Python 3.12
uv run main.py --agent=explorer --game=ls20   # offline; no API key needed
uv run pytest -q                     # 59 tests
```

`.env` is preconfigured for offline play (`OPERATION_MODE=offline`,
`ENVIRONMENTS_DIR=../environment_files`).

## Full sweep

```bash
cd ARC-AGI-3-Kaggle-Starter 2>/dev/null; cd ..   # from repo root
ARC-AGI-3-Agents/.venv/bin/python sweeps/sweep_adaptive.py   # ~2 min, 7 workers
```

## Kaggle submission

Uses the [official starter kit](https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter)
flow: `make setup` → `make play-local` → `make submit` (pushes a private notebook;
does **not** spend the daily submission) → click *Submit to Competition* on
kaggle.com → pick `submission.parquet`. The rerun plays the 110 hidden games
through a `gateway:8001` sidecar; the gateway computes the score file.

## Rules notes

- Evaluation offline → no API models; bundled open weights only (RTX 6000 pool available).
- Prize eligibility requires open-sourcing (CC-BY-4.0/OSI) — this repo is part of that.
- 1 submission/day, ≤ 9h notebook runtime, 2 milestones (Jun 30 / Sep 30, 2026),
  submissions close Nov 2, 2026.
