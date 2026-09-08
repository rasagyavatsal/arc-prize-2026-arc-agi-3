# Adaptive-Budget Sweep — Explorer Agent vs 25 Public Games

**Run:** 2026-09-08 | agent: `explorer` (deterministic, CPU-only) | policy: `budget = clamp(2.0 x baseline_total, 300, 2500)` + 420s wall-clock guard | 7 parallel workers | engine ran at ~90-220 fps locally, so budgets (not time) bound every game.

## Results (sorted by score)

| game | levels | human baseline | budget | actions taken | score % | note |
|------|-------|----------------|--------|---------------|---------|------|
| lp85 | 8 | 388 | 776 | 105 | **12.500** | completed levels(s) |
| cd82 | 6 | 171 | 342 | 342 | **3.309** | improved vs 80-action run (0.0) |
| sp80 | 6 | 518 | 1036 | 1036 | **1.087** | budget exhausted |
| m0r0 | 6 | 1107 | 2214 | 2214 | **0.357** | budget exhausted |
| ar25 | 8 | 748 | 1496 | 1496 | 0.000 | budget exhausted |
| g50t | 7 | 879 | 1758 | 1758 | 0.000 | budget exhausted |
| sk48 | 8 | 1070 | 2140 | 2140 | 0.000 | budget exhausted |
| ls20 | 7 | 776 | 1552 | 1552 | 0.000 | budget exhausted (sokoban variant) |
| cn04 | 6 | 789 | 1578 | 1578 | 0.000 | budget exhausted |
| ka59 | 7 | 730 | 1460 | 1460 | 0.000 | budget exhausted |
| re86 | 8 | 1255 | 2500 | 2500 | 0.000 | budget exhausted |
| lf52 | 10 | 1339 | 2500 | 2500 | 0.000 | budget exhausted |
| wa30 | 9 | 1843 | 2500 | 2500 | 0.000 | budget exhausted |
| sb26 | 8 | 213 | 426 | 426 | 0.000 | budget exhausted |
| sc25 | 6 | 350 | 700 | 700 | 0.000 | budget exhausted |
| tn36 | 7 | 317 | 634 | 634 | 0.000 | budget exhausted |
| tr87 | 6 | 414 | 828 | 828 | 0.000 | budget exhausted |
| r11l | 6 | 233 | 466 | 466 | 0.000 | budget exhausted |
| vc33 | 7 | 447 | 894 | 357 | 0.000 | early give-up |
| dc22 | 6 | 1228 | 2456 | 192 | 0.000 | early give-up (click-only) |
| bp35 | 9 | 651 | 1302 | 39 | 0.000 | early give-up (click-only) |
| tu93 | 9 | 462 | 924 | 25 | 0.000 | early give-up (click-only) |
| ft09 | 6 | 208 | 416 | 23 | 0.000 | early give-up (click-only) |
| s5i5 | 8 | 638 | 1276 | 16 | 0.000 | early give-up (click-only) |
| su15 | 9 | 361 | 722 | 13 | 0.000 | early give-up (click-only) |

**Total score (public 25): 0.69%** — same order of magnitude as frontier AI on the hidden set (~0.51%).

## Findings

1. **Adaptive budgets alone do not unlock levels.** 13/25 games burned their entire (2x-human) budget without completing a level. ls20, g50t, sk48, cn04, ka59, re86, lf52 need discovery insight, not exploration volume.
2. **Click-only games die from premature give-up, not budget.** In 6 click-only games the agent's color-centroid click probing never moves anything -> no-effect streak -> RESET loop -> honest give-up after ~6 attempts at only 2-10% of budget. The stopping heuristics are tuned for the old 80-action budget; with real budgets they quit way too early.
3. **Bigger budgets helped exactly the games where exploration suffices:** cd82 went 0.0 -> 3.31, lp85 12.5, sp80/m0r0 > 0. The memory-graph architecture works when probing can reach the frontier.
4. **Give-up logging was invisible** (named logger without handlers) - fixed in the debug run; sweep log itself has no give-up lines for that reason.
5. Verified locally: the engine imposes NO fps throttling offline (~200 fps), so 9h Kaggle wall-clock is not the binding constraint for a programmatic agent; per-game time limits only matter for slow LLM agents.

## Implications for next iteration

- Scale stopping heuristics with budget (give-up thresholds, probe caps, attempt caps) - quick win, likely flips the 6 click-only games off zero.
- Click targeting needs a better model than color centroids: probe interactable sprite cells (collidable/moving objects), not region centers.
- ls20-family games need semantic state understanding (shape matching) - candidate for a small VLM layer on top of the memory graph.
