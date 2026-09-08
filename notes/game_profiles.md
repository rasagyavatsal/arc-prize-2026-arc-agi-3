# ARC-AGI-3 Game Environment Profiles

Static analysis of 25 games in `environment_files/<game_id>/<hash>/` (metadata.json + obfuscated `<game_id>.py`). No game code was imported or executed.

## Headline numbers

- Games analyzed: **25**; total level definitions: **183**.
- Levels per game: **6–10** (mean 7.32, median 7); distribution: {'6': 9, '7': 5, '8': 6, '9': 4, '10': 1}.
- Level count from `.py` matches `len(baseline_actions)` for all 25 games (0 mismatches).
- Human baseline actions per game (sum over levels): **171–1843** (mean 685.4, median 638).
- Action-space size (distinct ACTION ids): distribution {'1': 6, '2': 1, '3': 1, '4': 3, '5': 6, '6': 5, '7': 3}.
- Input types: **click-only: 7**, **keyboard-only: 6**, **mixed: 12**; 19 of 25 games handle ACTION6 (click).
- Grid dims range 8–64; 23/25 games use only square grids; 2 games (dc22) use non-square (wide) grids up to 64x44+.

## Per-game table

| game | levels | actions used | tag | total baseline actions | baseline per level (min–max) | grid size (levels) | fps |
|------|--------|--------------|-----|------------------------|------------------------------|--------------------|-----|
| ar25 | 8 | `[1, 2, 3, 4, 5, 6, 7]` | keyboard_click | 748 | 32–233 | 21x21 (all) | 6 |
| bp35 | 9 | `[1, 2, 3, 4, 5, 6, 7]` | keyboard_click | 651 | 21–163 | 8x8 (all) | 20 |
| cd82 | 6 | `[1, 2, 3, 4, 5, 6]` | keyboard_click | 171 | 8–55 | 64x64 (all) | 30 |
| cn04 | 6 | `[1, 2, 3, 4, 5, 6]` | keyboard_click | 789 | 29–300 | 20x20 (all) | 5 |
| dc22 | 6 | `[1, 2, 3, 4, 6]` | keyboard_click | 1228 | 59–578 | 64x44 … 64x64 [3 distinct] | 15 |
| ft09 | 6 | `[6]` | (none) | 208 | 12–65 | 32x32 (all) | 8 |
| g50t | 7 | `[1, 2, 3, 4, 5]` | keyboard | 879 | 54–230 | 64x64 (all) | 30 |
| ka59 | 7 | `[1, 2, 3, 4, 6]` | keyboard_click | 730 | 28–326 | 45x45 … 63x63 [3 distinct] | 10 |
| lf52 | 10 | `[1, 2, 3, 4, 5, 6, 7]` | click | 1339 | 32–244 | 8x8 (all) | 30 |
| lp85 | 8 | `[6]` | click | 388 | 16–159 | 27x32 … 63x63 [8 distinct] | 20 |
| ls20 | 7 | `[1, 2, 3, 4]` | keyboard | 776 | 22–192 | 64x64 (all) | 30 |
| m0r0 | 6 | `[1, 2, 3, 4, 5, 6]` | keyboard_click | 1107 | 26–500 | 11x11 … 15x15 [3 distinct] | 7 |
| r11l | 6 | `[6]` | click | 233 | 22–52 | 64x64 (all) | 30 |
| re86 | 8 | `[1, 2, 3, 4, 5]` | keyboard_click | 1255 | 26–424 | 64x64 (all) | 25 |
| s5i5 | 8 | `[6]` | click | 638 | 20–162 | 64x64 (all) | 15 |
| sb26 | 8 | `[5, 6, 7]` | keyboard_click | 213 | 18–58 | 64x64 (all) | 30 |
| sc25 | 6 | `[1, 2, 3, 4, 6]` | keyboard_click | 350 | 6–143 | 64x64 (all) | 20 |
| sk48 | 8 | `[1, 2, 3, 4, 6, 7]` | keyboard_click | 1070 | 61–230 | 64x64 (all) | 30 |
| sp80 | 6 | `[1, 2, 3, 4, 5, 6]` | keyboard_click | 518 | 25–152 | 16x16 … 20x20 [2 distinct] | 10 |
| su15 | 9 | `[6, 7]` | click | 361 | 8–115 | 64x64 (all) | 20 |
| tn36 | 7 | `[6]` | click | 317 | 26–72 | 64x64 (all) | 3 |
| tr87 | 6 | `[1, 2, 3, 4]` | keyboard | 414 | 40–146 | 64x64 (all) | 10 |
| tu93 | 9 | `[1, 2, 3, 4]` | keyboard_click | 462 | 14–123 | 39x39 … 51x51 [5 distinct] | 30 |
| vc33 | 7 | `[6]` | click | 447 | 7–152 | 32x32 … 64x64 [4 distinct] | 20 |
| wa30 | 9 | `[1, 2, 3, 4, 5]` | keyboard | 1843 | 68–442 | 64x64 (all) | 5 |

## Notes on the table

- `actions used` = `available_actions=[...]` from the game constructor. For **bp35, cn04, lf52** the constructor omits it; their sets are inferred from `GameAction` usages (upper bound; cn04's `_get_valid_actions` override exposes ids 1–5 dynamically).
- Tag vs actions mismatches: `tu93`/`re86` are tagged `keyboard_click` but expose no ACTION6; `su15`/`lf52` tagged `click` but reference ACTION1–7 in code. Trust `available_actions` over metadata tags.

## Action semantics observed (from usage sites)

- ACTION1–4: directional / 4-way movement or selection (the common backbone; present in 18/25 games).
- ACTION5: context-specific secondary verb: rotate (ar25, g50t, re86), confirm/interact (cn04), pickup (wa30), undo-like special in sb26.
- ACTION6: click — handled via `action.data` (x,y coordinates) colliding with `sys_click`-tagged sprites; 19 games reference it, 7 games are click-only.
- ACTION7: undo / state restore (explicit undo stack in ar25, lf52; su15 exposes `[6,7]`).
- RESET: 6 games (bp35, dc22, g50t, lf52, sc25, sp80) reference `GameAction.RESET`.
- 13 games override `_get_valid_actions` to gate actions per state — valid action sets can be smaller than the advertised action space.

## Win/loss and progression structure

- All games subclass `ARCBaseGame`; progression is via `next_level()`; failure via `lose()`. Every game calls `next_level()` exactly once (single linear progression).
- 10 games attach a per-level `StepCounter` in level `data` — an explicit per-level action budget (e.g. tu93 20–60, ls20 42/level, dc22 128–1024). The other 15 games manage timing implicitly.
- Level `data` keys are mostly obfuscated sprite ids; `StepCounter` is the one recurring generic key.
- Two games are heavily transformed rather than sprite-table driven: **bp35** (62 classes, single placeholder sprite, procedural graph world with explicit `win()`/`lose()` on gem/spike landing) and **lf52** (68 classes, undo-manager machinery). Their `Level(...)` blocks understate real complexity (file sizes 4.6k and 5.9k lines).
- File sizes vary 780–41,446 lines (ka59 is the largest; most games 1k–3k lines).

## Games grouped by tag

- `(none)` (1): ft09
- `click` (7): lf52, lp85, r11l, s5i5, su15, tn36, vc33
- `keyboard` (4): g50t, ls20, tr87, wa30
- `keyboard_click` (13): ar25, bp35, cd82, cn04, dc22, ka59, m0r0, re86, sb26, sc25, sk48, sp80, tu93

## Games grouped by action set

- `[1, 2, 3, 4]` (3): ls20, tr87, tu93
- `[1, 2, 3, 4, 5]` (3): g50t, re86, wa30
- `[1, 2, 3, 4, 5, 6]` (4): cd82, cn04, m0r0, sp80
- `[1, 2, 3, 4, 5, 6, 7]` (3): ar25, bp35, lf52
- `[1, 2, 3, 4, 6]` (3): dc22, ka59, sc25
- `[1, 2, 3, 4, 6, 7]` (1): sk48
- `[5, 6, 7]` (1): sb26
- `[6]` (6): ft09, lp85, r11l, s5i5, tn36, vc33
- `[6, 7]` (1): su15

## Default FPS distribution

{'3': 1, '5': 2, '6': 1, '7': 1, '8': 1, '10': 3, '15': 2, '20': 5, '25': 1, '30': 8}

## Hardest-looking games by human baseline effort

- wa30: 1843 total baseline actions over 9 levels (205/level avg), actions `[1, 2, 3, 4, 5]`.
- lf52: 1339 total baseline actions over 10 levels (134/level avg), actions `[1, 2, 3, 4, 5, 6, 7]`.
- re86: 1255 total baseline actions over 8 levels (157/level avg), actions `[1, 2, 3, 4, 5]`.
- dc22: 1228 total baseline actions over 6 levels (205/level avg), actions `[1, 2, 3, 4, 6]`.
- m0r0: 1107 total baseline actions over 6 levels (184/level avg), actions `[1, 2, 3, 4, 5, 6]`.

## Agent-design takeaways

1. **Small, mostly fixed action space**: 1–7 discrete ids; typical effective set is 4–6. Click-only games collapse to a single ACTION6 whose parameter space is the grid (coordinate selection), so exploration must plan click targets, not just button order.
2. **6–10 levels per game, linear progression**; `baseline_actions` lengths give a free per-level difficulty signal (e.g. wa30 levels need 171–435 actions).
3. **Step budgets exist and matter**: 10 games hard-limit moves per level (20–1024); blind random exploration will time out on tight-budget games (tu93: 20–60 moves).
4. **Mixed input is the common case**: 12 games need both movement keys and coordinate clicks; a policy head must handle parameterized actions.
5. **Undo/ACTION7 and RESET in a third of games** enable safe exploration (revert bad states) — detect and exploit when available.
