"""Failure taxonomy: why Explorer does not solve each DEV level.

Plays every DEV level as its own episode, exactly like heldout_eval.py, and
records how each episode ended and where its actions went:

  end          solved | gave_up | budget | timeout
  kind         move (simple actions only) | click (ACTION6 only) | mixed
  states       distinct state signatures seen; growth_q = count at 25/50/75/100%
  last_new     fraction of the episode at which the last new state appeared
  untried      untried (state, action) pairs left in memory at the end
  noop_frac    actions whose transition changed nothing outside the status bar
  why          actions per decision branch (frontier probe, replay, walk, ...)

Failures are bucketed:
  A exhausted  gave up: every known probe tried, no frontier left
  B growing    budget ran out while new states were still appearing
  C plateau    states stopped appearing early; the rest of the budget was burned

DEV only: the held-out split is never diagnosed per level.

Usage (from the repo root):
  ARC-AGI-3-Agents/.venv/bin/python sweeps/dev_taxonomy.py [--budget 1500]
      [--workers N] [--games a,b] [--out notes/dev_taxonomy.json]
"""
import argparse, json, logging, os, statistics, sys, time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import heldout_eval as he  # sets the offline env and sys.path


def run(game: str, level: int, budget: int, time_limit_s: float) -> dict:
    logging.disable(logging.WARNING)
    from arc_agi import Arcade
    from arcengine import GameState
    from agents.templates.explorer import Explorer

    log, giveup = [], []

    class Diag(Explorer):
        def is_done(self, frames, latest):
            return latest.levels_completed >= 1 or super().is_done(frames, latest)

        def _give_up_now(self, why):
            if not self._give_up:
                giveup.append(why)
            super()._give_up_now(why)

        def choose_action(self, frames, latest):
            a = super().choose_action(frames, latest)
            reasoning = getattr(a, "reasoning", None)
            why = reasoning.get("why", "reset") if isinstance(reasoning, dict) else "reset"
            log.append({"sig": self.last_sig, "act": a.name, "why": why,
                        "state": latest.state.name, "world": self.world_changed})
            return a

    env = Arcade().make(game)
    env._game.set_level(level)
    env.reset()
    ag = Diag(card_id="taxonomy", game_id=env.environment_info.game_id, agent_name="taxonomy",
              ROOT_URL="", record=False, arc_env=env, budget=budget, time_limit_s=time_limit_s)
    t0 = time.time()
    ag.main()
    solved = ag.frames[-1].levels_completed >= 1
    n = len(log)

    seen, growth, last_new = set(), [], 0
    for i, e in enumerate(log):
        if e["sig"] and e["sig"] not in seen:
            seen.add(e["sig"])
            last_new = i
        growth.append(len(seen))
    at = lambda f: growth[min(n - 1, int(f * n))] if n else 0

    lvl = [s for s in ag.memory if s.split("|")[1:2] == ["0"]]
    acts = Counter(e["act"] for e in log)
    avail = sorted({a for node in ag.memory.values() for a in node["avail"]})
    kind = "click" if set(avail) <= {0, 6} else "move" if 6 not in avail else "mixed"
    if solved:
        end = "solved"
    elif giveup:
        end = "gave_up"
    elif ag.action_counter >= budget:
        end = "budget"
    else:
        end = "timeout"
    return {
        "game": game, "level": level, "solved": solved, "end": end,
        "giveup": giveup[0] if giveup else None, "actions": ag.action_counter,
        "kind": kind, "states": len(seen), "growth_q": [at(.25), at(.5), at(.75), at(1.0)],
        "last_new": round(last_new / max(1, n), 2),
        "untried": sum(len(ag._untried_at(s, True)) for s in lvl),
        "fine": 0 in ag.fine_levels,
        # world_changed on step i describes the action of step i-1
        "noop_frac": round(sum(not e["world"] for e in log[1:]) / max(1, n - 1), 2),
        "game_overs": sum(e["state"] == GameState.GAME_OVER.name for e in log),
        "resets": acts.get("RESET", 0),
        "click_frac": round(acts.get("ACTION6", 0) / max(1, n), 2),
        "why": dict(Counter(e["why"] for e in log).most_common()),
    }


def bucket(r: dict) -> str:
    if r["solved"]:
        return "solved"
    if r["end"] == "gave_up":
        return "A exhausted"
    return "B growing" if r["last_new"] >= 0.9 else "C plateau"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=1500)
    ap.add_argument("--time-limit", type=float, default=900.0)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--games", help="comma-separated subset of DEV")
    ap.add_argument("--out", default=os.path.join(he.ROOT, "notes", "dev_taxonomy.json"))
    args = ap.parse_args()

    games = he.DEV
    if args.games:
        asked = args.games.split(",")
        assert set(asked) <= set(he.DEV), "dev games only"
        games = [g for g in games if g in asked]
    eps = [(g, k) for g in games for k in range(len(he.load_meta(g)["baseline_actions"]))]
    recs = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run, g, k, args.budget, args.time_limit) for g, k in eps]
        for f in as_completed(futs):
            recs.append(f.result())
    recs.sort(key=lambda r: (r["game"], r["level"]))

    groups = defaultdict(list)
    for r in recs:
        groups[bucket(r)].append(r)
    for name in sorted(groups):
        rs = groups[name]
        kinds = Counter(r["kind"] for r in rs)
        print(f"{name}: {len(rs)} {dict(kinds)}  "
              f"median states {statistics.median(r['states'] for r in rs)}  "
              f"no-op {statistics.median(r['noop_frac'] for r in rs):.2f}")
        print("   " + " ".join(f"{r['game']}L{r['level']}" for r in rs))
    print("\nfailed-level action spend by kind:")
    for kind in ("move", "click", "mixed"):
        rs = [r for r in recs if not r["solved"] and r["kind"] == kind]
        if not rs:
            continue
        why = Counter()
        for r in rs:
            why.update(r["why"])
        total = sum(why.values())
        noop = sum(r["noop_frac"] * r["actions"] for r in rs) / sum(r["actions"] for r in rs)
        print(f"  {kind:5s} {len(rs):2d} levels | no-op {100 * noop:.0f}% | "
              + ", ".join(f"{k} {100 * v / total:.0f}%" for k, v in why.most_common(5)))

    with open(args.out, "w") as f:
        json.dump({"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "budget": args.budget,
                   "results": recs}, f, indent=1)


if __name__ == "__main__":
    main()
