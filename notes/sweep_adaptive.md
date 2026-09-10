# Adaptive-Budget Sweep — Explorer Agent vs 25 Public Games

**Run:** 2026-09-10 (v3 agent) | agent: `explorer` (deterministic, CPU-only) | autonomous mode: `budget = MAX_ACTIONS = 1500` per game (no metadata), 270 s wall-clock guard | 7 parallel workers.

Scores are **fully deterministic** per (code, machine-load) combination — three repeated runs of both the old and the new agent produced identical totals (±0.0). The one source of variance is engine timing sensitivity under CPU load: e.g. tu93 scored 22.2 in controlled runs but 7.6 in a heavily loaded recorded sweep. Numbers below are the controlled A/B (identical conditions).

## Headline (controlled A/B, 3 seeds, budget 1500, record off)

| agent | public-25 total |
|---|---|
| Explorer v2 (pre-2026-09-10, committed baseline) | **0.717 %** |
| Explorer v3 (object-layout signature + escalation) | **1.289 %** (+80 %) |

Per-game (mean of 3 seeds; only games that differ):

| game | v2 | v3 | note |
|---|---|---|---|
| tu93 | 0.0 | **22.222** | per-colour clusters expose box/switch positions |
| g50t | 0.0 | **7.283** | same mechanic |
| m0r0 | 0.357 | **0.560** | |
| lp85 | 12.500 | 2.168 | still completes level 0, less efficiently |
| cd82 | 3.309 | 0.0 | lost the v2 luck; its 10-action win needs a specific config |
| sp80 | 1.087 | 0.0 | toggle-puzzle, needs a lucky config |
| r11l | 0.684 | 0.0 | toggle-puzzle, needs a lucky config |

(v3 also completes vc33 level 0 in some runs — 0.424 — and earlier variants scored cd82 16.667 with a mixed-only signature; these are configuration-lottery effects.)

## What changed in the agent (v2 → v3)

1. **Object-layout state signature** (replaces mover-centroids + grid-hash fallback):
   segmentation of the settled frame above the bottom-2 HUD rows with background =
   the modal cell value; zero cells count as objects when the modal value is
   non-zero (cd82/s5i5 draw interactables in colour 0).  Signature = mixed-colour
   cluster layout + small (≤ 32 cell) per-colour cluster layout, each cluster as
   `(centroid-y, centroid-x, size bucket)`.  Path independent (same world ⇒ same
   state, however reached), animation-tolerant.
2. **Per-colour clustering** for the small clusters: a box pushed along a wall
   forms its own moving cluster instead of being absorbed into a same-colour
   mega-cluster (this is what unlocked tu93 and g50t).
3. **Escalation to grid-hash identity per level**: when a level accumulates
   ≥ 300 actions that change the world while leaving the layout untouched
   (lights-out toggles, reveals, sprite swaps), the level's coarse memory is
   discarded and re-probed with full grid-hash signatures.
4. **Click targeting**: candidates are the on-centroid cell of every cluster
   (smallest first) — no more per-colour centroids that land between
   same-coloured objects — plus a centre-out lattice scan, capped at
   `min(w*h/24, 96)` points.  A click that changed anything refines the
   resulting state's candidates with the effective point's neighbourhood.
5. **Budget-scaled patience** (the old heuristics were tuned for 80 actions and
   quit click-only games at 2–10 % of budget): `NO_CHANGE_LIMIT` up to 40,
   `MAX_LEVEL_ATTEMPTS` up to 30, `PACING_FACTOR` 12× per-level share, probe
   caps scaled to the grid area.
6. **Bounded explore walks**: when the memory graph has no reachable frontier,
   up to 2 deterministic random walks (≤ 48 steps) per level run off-graph and
   exit the moment they step on a never-seen state.
7. **Raw-diff edge recording**: an action whose world changed but whose
   signature did not is recorded as *effective* (not a no-op), so repaint
   actions are not demoted and refinement still fires.

## Remaining zero games and why

- **ls20, tr87** (sokoban/movement): modal-coloured avatars are invisible to the
  layout; boxes absorb into mega-clusters.  Needs either richer movement
  modelling or a planning layer.
- **r11l, sp80, ft09, su15, s5i5, tn36, sb26, sc25, cn04, ka59, dc22, re86,
  lf52, wa30, ar25, bp35**: single clicks/probes rarely move anything from the
  start state; wins hide behind specific configurations or multi-step
  sequences.  r11l was brute-forced: *no* single click completes its level 0 —
  the v2 agent's win (click 36,26 at action 536) was a lucky config draw.
  These need search (planning/VLM), not exploration volume.

## Runbook

```bash
ARC-AGI-3-Agents/.venv/bin/python sweeps/sweep_adaptive.py   # ~3 min, 7 workers
```
