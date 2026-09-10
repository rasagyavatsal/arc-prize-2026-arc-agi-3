"""ARC-AGI-3 Submission Readiness Gate.

Enforces pre-submission quality, autonomy, budget, and breadth requirements
before spending Kaggle's limited daily submission slots (1-5/day):

1. Code Unity: my_agent.py and explorer.py share a single source of truth.
2. Autonomy: Agent relies ONLY on live observation FrameData, zero metadata.json
   or baseline_actions cheating.
3. Time & Budget: Wall-clock guard (<4.9 min/game, scalable to 110 games in 9h)
   and realistic action budget (>80).
4. Breadth: Multi-category capability check ensuring >= 2 games achieve score > 0.
"""
from __future__ import annotations

import argparse
import ast
import filecmp
import importlib.util
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # ARC-AGI-3-Kaggle-Starter
WORKSPACE = ROOT.parent
AGENT_FILE = ROOT / "agent" / "my_agent.py"
CANONICAL_EXPLORER = WORKSPACE / "ARC-AGI-3-Agents" / "agents" / "templates" / "explorer.py"
VENDOR_DIR = ROOT / "vendor" / "ARC-AGI-3-Agents"

# Ensure framework is on path
if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))
elif (WORKSPACE / "ARC-AGI-3-Agents").exists():
    sys.path.insert(0, str(WORKSPACE / "ARC-AGI-3-Agents"))

from arcengine import FrameData, GameAction, GameState

logging.basicConfig(level=logging.INFO, format="[gate] %(message)s")
logger = logging.getLogger("gate")


def load_agent_class():
    spec = importlib.util.spec_from_file_location("gate_user_agent", AGENT_FILE)
    if not spec or not spec.loader:
        raise RuntimeError(f"Could not load agent from {AGENT_FILE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "MyAgent"):
        raise RuntimeError("agent/my_agent.py must define 'MyAgent'")
    return module.MyAgent


def check_code_unity() -> tuple[bool, str]:
    """Check 1: Single Source of Truth & Pipeline Unity."""
    if not AGENT_FILE.exists():
        return False, f"Missing {AGENT_FILE}"

    # Check canonical explorer if in monorepo
    if CANONICAL_EXPLORER.exists():
        if AGENT_FILE.is_symlink():
            target = (AGENT_FILE.parent / os.readlink(AGENT_FILE)).resolve()
            if target != CANONICAL_EXPLORER.resolve():
                return False, f"my_agent.py symlink points to {target}, expected {CANONICAL_EXPLORER}"
        elif not filecmp.cmp(AGENT_FILE, CANONICAL_EXPLORER, shallow=False):
            return False, (
                "Divergence detected: my_agent.py and explorer.py contents differ! "
                "Run `make sync-agent` to unify them."
            )

    agent_cls = load_agent_class()
    if not hasattr(agent_cls, "MAX_ACTIONS"):
        return False, "MyAgent does not define MAX_ACTIONS"

    # Verify build_notebook
    build_script = ROOT / "scripts" / "build_notebook.py"
    if build_script.exists():
        spec = importlib.util.spec_from_file_location("gate_build_nb", build_script)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            nb = mod.build()
            cells = nb.get("cells", [])
            if len(cells) < 4:
                return False, f"Built notebook has only {len(cells)} cells, expected >= 4"

    return True, "my_agent.py matches canonical Explorer; notebook pipeline verified"


def check_autonomy() -> tuple[bool, str]:
    """Check 2: Zero Cheating / Zero Leakage (Pure FrameData Autonomy)."""
    if not AGENT_FILE.exists():
        return False, f"Missing {AGENT_FILE}"

    content = AGENT_FILE.read_text()

    # Static AST and lexical analysis
    forbidden_tokens = ["metadata.json", "baseline_actions", "environment_files", "environment_info"]
    for token in forbidden_tokens:
        if token in content:
            return False, (
                f"Cheating / leakage detected: '{token}' found in agent source! "
                "Agent must rely ONLY on live observation FrameData."
            )

    parsed = ast.parse(content, filename=str(AGENT_FILE))
    for node in ast.walk(parsed):
        if isinstance(node, ast.Attribute) and node.attr in ("baseline_actions", "environment_info"):
            return False, f"Cheating detected: agent accesses '.{node.attr}' attribute!"

    # Dynamic isolation test: execute agent with missing/empty environment_info
    agent_cls = load_agent_class()
    test_agent = agent_cls(
        card_id="gate-test",
        game_id="autonomy_test",
        agent_name="gate-agent",
        ROOT_URL="http://localhost:8001",
        record=False,
        arc_env=None,
    )

    # Feed synthetic frames
    frame1 = FrameData(
        game_id="autonomy_test",
        frame=[[[0, 1], [1, 0]]],
        state=GameState.NOT_FINISHED,
        levels_completed=0,
        win_levels=7,
        guid="test-guid",
        available_actions=[1, 2, 6],
    )
    action = test_agent.choose_action([frame1], frame1)
    if not isinstance(action, GameAction):
        return False, f"choose_action returned invalid action type: {type(action)}"

    if getattr(test_agent, "total_levels", None) != 7:
        return False, (
            f"Agent failed to derive total_levels from FrameData.win_levels "
            f"(got {getattr(test_agent, 'total_levels', None)}, expected 7)"
        )

    return True, "Agent is 100% autonomous: zero metadata/baseline_actions leakage"


def check_time_and_budget() -> tuple[bool, str]:
    """Check 3: Time & Budget Safety Guard."""
    agent_cls = load_agent_class()
    max_actions = getattr(agent_cls, "MAX_ACTIONS", None)
    time_limit_s = getattr(agent_cls, "TIME_LIMIT_S", None)

    if max_actions is None:
        return False, "MAX_ACTIONS not defined on MyAgent"
    if max_actions < 300:
        return False, (
            f"MAX_ACTIONS = {max_actions} is dangerously small! "
            "Kaggle sample had 80 which scored only 0.08 due to early cutoff. "
            "Set MAX_ACTIONS >= 300 (recommended 1500)."
        )
    if max_actions > 5000:
        return False, f"MAX_ACTIONS = {max_actions} is too high (>5000) for CPU quota."

    if time_limit_s is None:
        return False, "TIME_LIMIT_S not defined on MyAgent"
    if time_limit_s <= 0:
        return False, f"TIME_LIMIT_S = {time_limit_s}s must be strictly positive!"

    # Strict per-game limit: 9 hours / 110 games = 294.5s max
    if time_limit_s > 280.0:
        return False, (
            f"TIME_LIMIT_S = {time_limit_s}s exceeds safe ceiling of 280s! "
            "110 games * 280s = 8.55h < 9h notebook limit."
        )

    # Active wall-clock timeout test
    test_agent = agent_cls(
        card_id="gate-test",
        game_id="timeout_test",
        agent_name="gate-agent",
        ROOT_URL="http://localhost:8001",
        record=False,
        arc_env=None,
    )
    f = FrameData(
        game_id="timeout_test",
        frame=[[[0]]],
        state=GameState.NOT_FINISHED,
        levels_completed=0,
        win_levels=5,
    )
    # Simulate timer past timeout limit
    test_agent.timer = time.time() - (test_agent.TIME_LIMIT_S + 5)
    if not test_agent.is_done([], f):
        return False, "is_done() failed to stop agent when wall-clock timer exceeded TIME_LIMIT_S!"

    worst_case_hours = round(110 * time_limit_s / 3600, 2)
    return True, (
        f"MAX_ACTIONS={max_actions}, TIME_LIMIT_S={time_limit_s}s "
        f"(110 games worst-case: {worst_case_hours}h < 9.0h)"
    )


def check_breadth(game_steps: int = 400, min_nonzero: int = 2) -> tuple[bool, str, list[dict]]:
    """Check 4: Breadth & Generalization Check across categories."""
    import arc_agi
    from arc_agi import OperationMode

    env_dir = None
    if (ROOT / "environment_files").exists():
        env_dir = str(ROOT / "environment_files")
    elif (WORKSPACE / "environment_files").exists():
        env_dir = str(WORKSPACE / "environment_files")

    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL, environments_dir=env_dir)
    agent_cls = load_agent_class()

    # Representative evaluation battery across input categories:
    # g50t = keyboard, lp85 = click, tu93 = keyboard_click
    candidate_games = ["g50t", "lp85", "tu93"]
    results = []
    nonzero_count = 0

    print(f"\n[gate] Running Breadth Check on {len(candidate_games)} diverse games (max {game_steps} steps each)...")

    for gid in candidate_games:
        t0 = time.time()
        card_id = arc.open_scorecard(tags=[f"gate-{gid}"])
        env = arc.make(gid, scorecard_id=card_id)
        if env is None:
            results.append({"game": gid, "score": 0.0, "levels": 0, "actions": 0, "elapsed": 0.0, "status": "ERROR"})
            arc.close_scorecard(card_id)
            continue

        agent = agent_cls(
            card_id=card_id,
            game_id=gid,
            agent_name=f"GateAgent.{gid}",
            ROOT_URL="http://localhost",
            record=False,
            arc_env=env,
            budget=game_steps,
        )
        agent.main()
        sc = arc.close_scorecard(card_id)
        elapsed = round(time.time() - t0, 2)
        final = agent.frames[-1] if agent.frames else None
        levels = agent.levels_completed
        actions = agent.action_counter

        # Calculate isolated game score strictly from this game's scorecard
        score = 0.0
        if sc and sc.environments:
            for env_s in sc.environments:
                if gid in env_s.id:
                    score = float(env_s.score)
                    levels = max(levels, env_s.levels_completed)
        elif sc and hasattr(sc, "score") and sc.score is not None:
            score = float(sc.score)

        if score > 0.0 or levels > 0:
            nonzero_count += 1
            status = "PASS (Score > 0)"
        else:
            status = "ZERO"

        results.append({
            "game": gid,
            "score": round(score, 3),
            "levels": levels,
            "actions": actions,
            "elapsed": elapsed,
            "status": status,
        })
        print(f"  → {gid:6}: score={round(score, 3):<6} levels={levels} actions={actions:<4} time={elapsed}s [{status}]")

    passed = nonzero_count >= min_nonzero
    summary_msg = (
        f"Non-zero scoring games: {nonzero_count}/{len(candidate_games)} "
        f"(required >= {min_nonzero})"
    )
    return passed, summary_msg, results


def run_gate(quick: bool = False, skip_breadth: bool = False) -> bool:
    print("=" * 70)
    print("        ARC-AGI-3 SUBMISSION READINESS GATE (Pre-Submit Check)        ")
    print("=" * 70)

    stages = [
        ("Code Unity & Pipeline", check_code_unity),
        ("Autonomous Check (Zero Leaks)", check_autonomy),
        ("Time & Budget Safety Guard", check_time_and_budget),
    ]

    all_passed = True

    for name, check_fn in stages:
        ok, msg = check_fn()
        status_tag = "[PASS]" if ok else "[FAIL]"
        print(f"{status_tag} {name}: {msg}")
        if not ok:
            all_passed = False
            break

    breadth_results = []
    if all_passed and not skip_breadth:
        steps = 300 if quick else 400
        ok, msg, breadth_results = check_breadth(game_steps=steps, min_nonzero=2)
        status_tag = "[PASS]" if ok else "[FAIL]"
        print(f"\n{status_tag} Breadth Check: {msg}")
        if not ok:
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print(">>> GATE PASSED: Submission is verified ready for Kaggle! <<<")
        print("=" * 70)
        return True
    else:
        print(">>> GATE FAILED: Submission blocked to protect Kaggle daily quota! <<<")
        print("Fix the issues above before running `make submit`.")
        print("=" * 70)
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARC-AGI-3 Submission Readiness Gate")
    parser.add_argument("--quick", action="store_true", help="Run fast breadth evaluation")
    parser.add_argument("--skip-breadth", action="store_true", help="Skip live game breadth evaluation")
    args = parser.parse_args()

    success = run_gate(quick=args.quick, skip_breadth=args.skip_breadth)
    sys.exit(0 if success else 1)
