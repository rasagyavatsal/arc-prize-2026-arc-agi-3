"""Held-out evaluation: play every level of a fixed game split as its own episode.

The public-25 sweep over-predicts the leaderboard (0.71% local vs 0.09 on the
110 hidden games): almost all of its points come from tu93/g50t/lp85, which
the agent was tuned on.  This harness scores the agent on games kept out of
tuning, and starts it at *every* level (not just level 0) so 15 held-out games
give ~100 independent episodes instead of 15 near-zero game scores.

Splits (fixed - do not move games between them, or the held-out number stops
meaning anything):
  DEV      games that drove past tuning decisions (named in notes, commits,
           tests or the agent) plus ls20, the default demo game.  Tune here.
  HELDOUT  games only ever seen as zeros in a bulk list.  Measure only; do
           not read their per-level results to pick heuristics.

Per episode: RESETs restart the current level only (ONLY_RESET_LEVELS), the
agent gets a fresh memory and the Kaggle budget, and the episode ends when the
level is solved or the budget runs out.  baseline_actions are read here for
scoring only; the agent never sees them.

Reported per split:
  solved        levels solved / levels played (primary signal)
  level-0       solved first levels (what dominates the leaderboard today)
  linear        mean min(100, 100*baseline/actions)   - local scorecard formula
  squared       mean min(1.15, (baseline/actions)^2)  - competition formula (unweighted)

Usage (from the repo root):
  ARC-AGI-3-Agents/.venv/bin/python sweeps/heldout_eval.py [--split heldout|dev|all]
      [--budget 1500] [--workers N] [--games a,b]
"""
import argparse, glob, json, logging, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = "/Users/rasagyavatsal/arc-prize-2026-arc-agi-3"
REPO = os.path.join(ROOT, "ARC-AGI-3-Agents")
os.environ["OPERATION_MODE"] = "offline"
os.environ["ENVIRONMENTS_DIR"] = os.path.join(ROOT, "environment_files")
os.environ["ONLY_RESET_LEVELS"] = "true"
sys.path.insert(0, REPO)

DEV = ["cd82", "g50t", "lp85", "ls20", "m0r0", "r11l", "s5i5", "sp80", "tu93", "vc33"]
HELDOUT = ["ar25", "bp35", "cn04", "dc22", "ft09", "ka59", "lf52", "re86", "sb26",
           "sc25", "sk48", "su15", "tn36", "tr87", "wa30"]
SPLITS = {"dev": DEV, "heldout": HELDOUT, "all": DEV + HELDOUT}


def load_meta(game: str) -> dict:
    for p in glob.glob(os.path.join(ROOT, "environment_files", game, "*", "metadata.json")):
        return json.load(open(p))
    return {}


def play_level(game: str, level: int, budget: int, time_limit_s: float) -> dict:
    logging.disable(logging.WARNING)
    from arc_agi import Arcade
    from arcengine import FrameData
    from agents.templates.explorer import Explorer

    class LevelExplorer(Explorer):
        def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
            return latest_frame.levels_completed >= 1 or super().is_done(frames, latest_frame)

    baseline = load_meta(game)["baseline_actions"][level]
    rec = {"game": game, "level": level, "baseline": baseline, "solved": False,
           "actions": None, "error": None}
    t0 = time.time()
    try:
        env = Arcade().make(game)
        env._game.set_level(level)
        env.reset()  # level reset at `level`, so the first observation is that level
        agent = LevelExplorer(
            card_id="heldout", game_id=env.environment_info.game_id, agent_name="heldout",
            ROOT_URL="", record=False, arc_env=env, budget=budget, time_limit_s=time_limit_s,
        )
        agent.main()
        rec["actions"] = agent.action_counter
        rec["solved"] = agent.frames[-1].levels_completed >= 1
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
    rec["elapsed_s"] = round(time.time() - t0, 1)
    if rec["solved"]:
        ratio = baseline / max(1, rec["actions"])
        rec["linear"] = min(100.0, 100 * ratio)
        rec["squared"] = min(1.15, ratio ** 2)
    else:
        rec["linear"] = rec["squared"] = 0.0
    return rec


def summarize(name: str, recs: list[dict]) -> dict:
    n = len(recs)
    lvl0 = [r for r in recs if r["level"] == 0]
    return {
        "split": name, "levels": n,
        "solved": sum(r["solved"] for r in recs),
        "level0_solved": sum(r["solved"] for r in lvl0), "games": len(lvl0),
        "linear_mean": round(sum(r["linear"] for r in recs) / max(1, n), 3),
        "squared_mean_pct": round(100 * sum(r["squared"] for r in recs) / max(1, n), 3),
        "errors": sum(r["error"] is not None for r in recs),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=sorted(SPLITS), default="heldout")
    ap.add_argument("--games", help="comma-separated subset of the split")
    ap.add_argument("--budget", type=int, default=1500, help="actions per level (Kaggle per-game budget)")
    ap.add_argument("--time-limit", type=float, default=900.0, help="seconds per level")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", help="results JSON (default notes/heldout_eval_<split>.json)")
    args = ap.parse_args()

    games = SPLITS[args.split]
    if args.games:
        games = [g for g in games if g in args.games.split(",")]
    episodes = [(g, k) for g in games for k in range(len(load_meta(g)["baseline_actions"]))]
    print(f"{args.split}: {len(games)} games, {len(episodes)} levels | budget={args.budget} "
          f"| workers={args.workers}", flush=True)

    recs, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(play_level, g, k, args.budget, args.time_limit) for g, k in episodes]
        for f in as_completed(futs):
            r = f.result()
            recs.append(r)
            if r["solved"] or r["error"]:
                print(f"  {r['game']} L{r['level']}: "
                      + (f"ERROR {r['error']}" if r["error"] else
                         f"solved in {r['actions']} (baseline {r['baseline']})"), flush=True)
    recs.sort(key=lambda r: (r["game"], r["level"]))

    by_game = {}
    for r in recs:
        by_game.setdefault(r["game"], []).append(r)
    print("\ngame   solved levels")
    for g, rs in by_game.items():
        print(f"{g}   {sum(r['solved'] for r in rs)}/{len(rs)}  "
              + "".join("#" if r["solved"] else "." for r in rs))

    summary = summarize(args.split, recs)
    print(f"\n{summary['split']}: solved {summary['solved']}/{summary['levels']} levels "
          f"| level-0 {summary['level0_solved']}/{summary['games']} games "
          f"| linear {summary['linear_mean']} | squared {summary['squared_mean_pct']}% "
          f"| errors {summary['errors']} | {round((time.time() - t0) / 60, 1)} min", flush=True)

    out = args.out or os.path.join(ROOT, "notes", f"heldout_eval_{args.split}.json")
    with open(out, "w") as f:
        json.dump({"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "budget": args.budget,
                   "summary": summary, "results": recs}, f, indent=1)


if __name__ == "__main__":
    main()
