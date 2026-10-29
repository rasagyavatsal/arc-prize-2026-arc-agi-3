# ARC Prize 2026 — ARC-AGI-3 Agent

Agent development for the [ARC Prize 2026 - ARC-AGI-3](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3)
Kaggle competition ($850K track): build an AI agent that plays 110 hidden,
instruction-free game environments — exploring, goal-finding, and winning levels
with human-competitive action efficiency.

## Current status

| Milestone | Result |
|---|---|
| Explorer v3 (object-layout signature, escalation, scaled patience) | **0.71%** on the 25 public games, all run concurrently (the 1.29% controlled run in `notes/sweep_adaptive.md` does not reproduce) |
| Leaderboard | **0.09** (2026-09-10 and 2026-10-27). Removing the 270s per-game cap did not change it: the agent does not generalize to unseen games |
| Held-out evaluation (`sweeps/heldout_eval.py`) | **4/113** levels solved on 15 held-out games (v3: 0/113), dev **14/70** (v3: 7/70) |
| Explorer v4 (inferred status-bar mask, per-colour click outcomes) | held-out 0 → 2 levels; dev 7 → 11 |
| Explorer v4.1 (full probes for click-reached states, inert click targets, RESET-loop fix) | held-out 2 → **4** levels; dev 11 → 14; leaderboard not yet resubmitted |
| LLM adviser (GPT-OSS-120B via vLLM on the RTX Pro 6000, issue #7) | dev 14 → 16 (within noise; a no-model control gets 15); held-out 4 → **3 and 2**: not submitted, off by default |
| Kaggle submission pipeline | **validated end-to-end** (official starter kit, notebook pushed) |
| Submission Readiness Gate | 4 checks (code unity, autonomy, budget, breadth) — passing |
| Game intelligence | all 25 public games statically profiled |

Scoring (RHAE): per completed level `(human_baseline / agent_actions)^2` capped at 1.15,
weighted by level index; total = mean over games. Leaderboard top ≈ 71% (2026-10-27). Scores are
deterministic per machine-load; engine timing sensitivity gives ±0.6% run-to-run
spread on the public set (see `notes/sweep_adaptive.md`).

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
ARC-AGI-3-Agents/.venv/bin/python sweeps/sweep_adaptive.py   # ~4 min, all 25 games concurrent (like Kaggle)
```

## Held-out evaluation

The public-25 score over-predicts the leaderboard because its points come from
games the agent was tuned on. `sweeps/heldout_eval.py` keeps a fixed split of
10 dev games (tune on these) and 15 held-out games (measure only; don't pick
heuristics from their per-level results), and starts the agent at every level
so each split yields one episode per level.

```bash
ARC-AGI-3-Agents/.venv/bin/python sweeps/heldout_eval.py --split heldout   # ~2.5 min, 113 levels
ARC-AGI-3-Agents/.venv/bin/python sweeps/heldout_eval.py --split dev       # ~1.5 min, 70 levels
ARC-AGI-3-Agents/.venv/bin/python sweeps/dev_taxonomy.py                   # why each dev level fails
```

Baseline (Explorer v3, budget 1500/level, 2026-10-27): held-out **0/113** levels,
dev **7/70**. Explorer v4 (2026-10-29): held-out **2/113**, dev **11/70**.
Explorer v4.1 (2026-10-29): held-out **4/113** (level-0 2/15, squared 0.011%),
dev **14/70** (squared 0.909%). Keeping a move's real result when a lost life
sends the player back to the start (2026-10-29) leaves both solve counts
unchanged but ends the ls20 L1/L2 plateaus (`notes/dev_taxonomy.json`)
(results in `notes/heldout_eval_*.json`). The held-out solve count is the
number to move.

## LLM adviser (issue #7)

A local open-weight model (GPT-OSS-120B behind vLLM; no internet on Kaggle)
can reorder the Explorer's frontier. It never chooses actions itself. Set
`ADVISER_URL` (an OpenAI-compatible `/v1` endpoint) to enable it; leave it
unset for plain Explorer behaviour.

- **When:** at the start of each level, and again after `ADVISER_PLATEAU`
  actions with no new state. At most 3 calls per level.
- **Prompt:** an object list (colour, size, box), the status-bar mask, what
  each move changed, and click outcomes per colour. The raw grid is not sent.
- **Answer:** the likely player colours, goal objects, objects to click first
  and moves to try first. These only reorder untried probes; replay, memory
  and give-up logic are unchanged.
- **Fallback:** a timeout (`ADVISER_TIMEOUT_S`, default 30s), an unusable
  answer, a spent per-game budget (`ADVISER_BUDGET_S`, default 240s), the
  last 30 min before the run deadline, or a server still loading all mean
  plain Explorer order. The adviser is dropped after 3 failures in a row.
- **Tests:** `tests/unit/test_adviser.py` covers this with a scripted
  stand-in model and a stub HTTP server, so no GPU is needed.

Without `ADVISER_URL`, dev is unchanged at 14/70 (squared 0.909%). Wired to
a local stand-in server that answers "smallest objects first", dev ran 168
calls (~2.8k chars per prompt) with no errors.

GPU dev run: `python scripts/build_adviser_dev_notebook.py` (from
`ARC-AGI-3-Kaggle-Starter/`) writes a private RTX Pro 6000 notebook. It runs
dev without the adviser, starts vLLM, benchmarks latency and throughput,
then runs dev with the adviser. It takes about 8 minutes in total.

The first GPU run (2026-10-29) is recorded in `notes/adviser_dev_gpu.json`:

| | dev solved | level-0 | squared |
|---|---|---|---|
| without adviser | 14/70 | 9/10 | 0.909% |
| with GPT-OSS-120B | 14/70 | 10/10 | **0.998%** |

- **Server:** vLLM was healthy 251s after start. Throughput was 38 calls/min
  at 1 concurrent call and 993 calls/min at 64, with p50 latency of
  1.6–3.0s.
- **Prompt size:** about 600–790 prompt tokens and 110 completion tokens.
- **Adviser on dev:** 169 calls, 143 usable answers, 4.1s per call.
- **Gains:** s5i5 L0 is newly solved, r11l L0 went from 163 to 87 actions,
  and vc33 L1 from 519 to 137.
- **Loss:** s5i5 L3 is no longer solved.
- **Unsure answers:** the remaining 26 calls returned valid JSON with every
  list empty. Since this run, such answers no longer count as failures.

**Dev tuning** (one GPU session, `notes/adviser_dev_tuning.json`, solved /
squared):

| variant | solved | squared |
|---|---|---|
| baseline | 14 | 0.909% |
| no-model control (`sweeps/standin_adviser.py`) | 15 | 1.038% |
| v1 | 13 | 0.909% |
| v1 with medium reasoning | 13 | 1.160% |
| v2 | 15 | 0.797% |
| v2 with the early call | 15 | 0.803% |
| v2 with medium reasoning and the early call | 16 | 0.982% |

Model runs vary by about one level from run to run. The last variant
solved the most levels, so it became the default.

**Held-out** (`notes/adviser_heldout.json`, summary lines only):

| | solved | squared |
|---|---|---|
| baseline | 4/113 | 0.011% |
| control | 4/113 | 0.011% |
| adviser | 3/113 | 0.013% |
| adviser, repeated | 2/113 | 0.018% |

**Verdict (2026-10-29): the adviser does not generalize.** It solves fewer
held-out levels, so it is not submitted and the submission builder keeps
`ADVISER = False`. Its dev gains were within run-to-run noise. A no-model
control (preferring the colours of the smallest objects when clicking)
matched it on dev and scored the same as the baseline on held-out. The code stays in place, opt-in through
`ADVISER_URL`, for any later attempt.

Kaggle setup needed for vLLM:

- **Accelerator:** `NvidiaRtxPro6000`. Any other name silently runs on a T4.
- **Image:** the official template's Python 3.12 image, because the
  vllm-deps wheels are built for cp312.
- **Driver path:** `/usr/local/nvidia/lib64` added to `LD_LIBRARY_PATH`.
- **Torch:** `torch==2.10.0+cu128` named explicitly. Otherwise the image's
  `+cpu` build satisfies vLLM's requirement and stays installed.

## Kaggle submission

Uses the [official starter kit](https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter)
flow: `make setup` → `make play-local` → `make submit` (pushes a private notebook;
does **not** spend the daily submission) → click *Submit to Competition* on
kaggle.com → pick `submission.parquet`. The rerun plays the 110 hidden games
through a `gateway:8001` sidecar; the gateway computes the score file.

## Rules notes

- Evaluation offline → no API models; bundled open weights only (RTX 6000 pool available).
- Prize eligibility requires open-sourcing before private scores are released:
  own code under a permissive public-domain licence (CC0 or MIT-0), third-party
  code under at least a shareable open-source licence — this repo is part of that.
- 1 submission/day, ≤ 9h notebook runtime, 2 milestones (Jun 30 / Sep 30, 2026),
  submissions close Nov 2, 2026.

## License

Our code is released under [MIT-0](LICENSE). Third-party parts keep their own
terms: the `ARC-AGI-3-Agents/` framework is MIT (ARC Prize, see its `LICENSE`;
our additions in it, such as `agents/templates/explorer.py`, are MIT-0), the
`ARC-AGI-3-Kaggle-Starter/` kit is ARC Prize's, and `environment_files/` are the
competition's game environments.
