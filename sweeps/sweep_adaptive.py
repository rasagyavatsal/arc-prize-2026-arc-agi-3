"""Adaptive-budget full sweep of the 25 public ARC-AGI-3 games with the Explorer agent.

Run from anywhere: it chdirs into the repo. Env overrides are set BEFORE arc_agi import
(arc_agi loads .env without override, so pre-set values win).

Budget policy (no official fixed budget exists - agent-side knob):
  budget(game) = clamp(round(2.0 * sum(baseline_actions)), 300, 2500)
  + the agent's own wall-clock guards (override with SWEEP_TIME_LIMIT_S).

All games run concurrently by default (SWEEP_WORKERS=0), like Kaggle's Swarm:
the scores then include the CPU contention the agent sees on the leaderboard.
"""
import os, sys, time, json, glob, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.join(ROOT, "ARC-AGI-3-Agents")
os.chdir(REPO)
RECORDINGS = os.path.join(ROOT, "sweep_recordings")
os.environ["RECORDINGS_DIR"] = RECORDINGS
os.makedirs(RECORDINGS, exist_ok=True)
sys.path.insert(0, REPO)

TIME_LIMIT_S = os.environ.get("SWEEP_TIME_LIMIT_S")
BUDGET_MULT, BUDGET_MIN, BUDGET_MAX = 2.0, 300, 2500
WORKERS = int(os.environ.get("SWEEP_WORKERS", "0"))  # 0 = all games at once

from arc_agi import Arcade  # noqa: E402
from agents.templates.explorer import Explorer  # noqa: E402

TIME_LIMIT_S = float(TIME_LIMIT_S) if TIME_LIMIT_S else Explorer.DEFAULT_TIME_LIMIT_S


def budget_for(baseline_total: int) -> int:
    return int(min(BUDGET_MAX, max(BUDGET_MIN, round(BUDGET_MULT * baseline_total))))


# Unified with canonical Explorer: Explorer now has native budget & wall-clock guards
AdaptiveExplorer = Explorer
AUTONOMOUS_MODE = os.environ.get("SWEEP_AUTONOMOUS", "1") == "1"


def load_meta(base_gid: str) -> dict:
    for p in glob.glob(os.path.join(ROOT, "environment_files", base_gid, "*", "metadata.json")):
        return json.load(open(p))
    return {}


def run_game(gid: str) -> dict:
    base = gid.split("-")[0]
    meta = load_meta(base)
    baselines = meta.get("baseline_actions", [])
    if AUTONOMOUS_MODE:
        # Evaluates exact autonomous submission behavior without metadata cheating
        budget = int(os.environ.get("MAX_ACTIONS", str(Explorer.DEFAULT_MAX_ACTIONS)))
    else:
        budget = budget_for(sum(baselines)) if baselines else Explorer.DEFAULT_MAX_ACTIONS
    t0 = time.time()
    arc = Arcade()
    card = arc.open_scorecard(tags=["sweep-adaptive"])
    rec = {
        "game": base, "full_id": gid, "levels_total": len(baselines),
        "baseline_total": sum(baselines), "budget": budget,
        "time_limit_s": TIME_LIMIT_S, "time_capped": False,
        "error": None,
    }
    try:
        env = arc.make(gid, scorecard_id=card)
        agent = Explorer(
            card_id=card, game_id=gid, agent_name="explorer-adaptive",
            ROOT_URL="", record=True, arc_env=env, budget=budget,
            time_limit_s=TIME_LIMIT_S,
        )
        agent.main()
        sc = arc.close_scorecard(card)
        sd = sc.model_dump() if sc else {}
        rec.update({
            "score_raw": sd.get("score"),
            "score_capped100": min(100.0, sd["score"]) if sd.get("score") is not None else None,
            "levels_completed": sd.get("levels_completed"),
            "actions": sd.get("actions"),
            "resets": sd.get("resets"),
            "state": str(sd.get("state")) if sd.get("state") is not None else None,
            "level_scores": sd.get("level_scores"),
        })
    except Exception:
        rec["error"] = traceback.format_exc(limit=4)
        try:
            arc.close_scorecard(card)
        except Exception:
            pass
    rec["elapsed_s"] = round(time.time() - t0, 1)
    rec["time_capped"] = rec["elapsed_s"] >= TIME_LIMIT_S - 5 and not rec.get("error")
    print(f"[done] {base}: levels {rec.get('levels_completed')}/{rec['levels_total']} "
          f"score {rec.get('score_raw')} actions {rec.get('actions')}/{rec['budget']} "
          f"({rec['elapsed_s']}s)", flush=True)
    return rec


def main() -> None:
    arc = Arcade()
    gids = sorted(getattr(e, "game_id") for e in arc.available_environments)
    workers = WORKERS or len(gids)
    print(f"sweeping {len(gids)} games | workers={workers} | time_limit={TIME_LIMIT_S}s", flush=True)
    results, t0 = [], time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_game, g): g for g in gids}
        for f in as_completed(futs):
            results.append(f.result())
    results.sort(key=lambda r: (-(r.get("score_capped100") or 0), r["game"]))
    total = sum(r.get("score_capped100") or 0 for r in results) / len(results)
    out = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "policy": {
        "budget": f"clamp({BUDGET_MULT}*baseline, {BUDGET_MIN}, {BUDGET_MAX})",
        "time_limit_s": TIME_LIMIT_S, "workers": workers,
    }, "total_score_public25": round(total, 3), "results": results}
    with open(os.path.join(ROOT, "notes", "sweep_adaptive.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(f"ALL DONE in {round((time.time()-t0)/60,1)} min | public-25 total score: {round(total,2)}%", flush=True)


if __name__ == "__main__":
    main()
