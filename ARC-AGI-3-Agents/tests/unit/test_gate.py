"""Unit tests for the Submission Readiness Gate logic."""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from arcengine import FrameData, GameState

import sys
ROOT = Path(__file__).resolve().parents[2]  # ARC-AGI-3-Agents
WORKSPACE = ROOT.parent
STARTER = WORKSPACE / "ARC-AGI-3-Kaggle-Starter"
sys.path.insert(0, str(STARTER))

from scripts.gate import (
    check_autonomy,
    check_code_unity,
    check_time_and_budget,
    load_agent_class,
)


@pytest.mark.unit
class TestGateVerification:
    def test_current_code_unity_passes(self):
        ok, msg = check_code_unity()
        assert ok is True, f"Code unity failed: {msg}"

    def test_current_autonomy_passes(self):
        ok, msg = check_autonomy()
        assert ok is True, f"Autonomy check failed: {msg}"

    def test_current_time_and_budget_passes(self):
        ok, msg = check_time_and_budget()
        assert ok is True, f"Time and budget check failed: {msg}"

    def test_autonomy_detects_cheating_metadata(self, monkeypatch):
        # Create a temporary agent file containing forbidden metadata reference
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_agent = Path(tmpdir) / "my_agent.py"
            fake_agent.write_text("import os\n# Cheating test\nmeta = 'metadata.json'\n")
            monkeypatch.setattr("scripts.gate.AGENT_FILE", fake_agent)
            ok, msg = check_autonomy()
            assert ok is False
            assert "metadata.json" in msg

    def test_autonomy_detects_baseline_actions_access(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_agent = Path(tmpdir) / "my_agent.py"
            fake_agent.write_text("class MyAgent:\n    def get_info(self, env):\n        return env.baseline_actions\n")
            monkeypatch.setattr("scripts.gate.AGENT_FILE", fake_agent)
            ok, msg = check_autonomy()
            assert ok is False
            assert "baseline_actions" in msg

    def test_budget_check_fails_on_small_actions(self, monkeypatch):
        agent_cls = load_agent_class()
        # Mock class with old 80 actions default
        class StubAgent(agent_cls):
            MAX_ACTIONS = 80
            TIME_LIMIT_S = 270.0

        monkeypatch.setattr("scripts.gate.load_agent_class", lambda: StubAgent)
        ok, msg = check_time_and_budget()
        assert ok is False
        assert "dangerously small" in msg

    def test_budget_check_fails_on_excessive_timeout(self, monkeypatch):
        agent_cls = load_agent_class()
        # Mock class with timeout > 280s (violates 9h / 110 games)
        class StubAgent(agent_cls):
            MAX_ACTIONS = 1500
            TIME_LIMIT_S = 350.0

        monkeypatch.setattr("scripts.gate.load_agent_class", lambda: StubAgent)
        ok, msg = check_time_and_budget()
        assert ok is False
        assert "exceeds safe ceiling" in msg

    def test_autonomy_detects_environment_info_access(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_agent = Path(tmpdir) / "my_agent.py"
            fake_agent.write_text("class MyAgent:\n    def get_info(self, env):\n        return env.environment_info\n")
            monkeypatch.setattr("scripts.gate.AGENT_FILE", fake_agent)
            ok, msg = check_autonomy()
            assert ok is False
            assert "environment_info" in msg

    def test_budget_check_fails_on_non_positive_timeout(self, monkeypatch):
        agent_cls = load_agent_class()
        class StubAgent(agent_cls):
            MAX_ACTIONS = 1500
            TIME_LIMIT_S = 0.0

        monkeypatch.setattr("scripts.gate.load_agent_class", lambda: StubAgent)
        ok, msg = check_time_and_budget()
        assert ok is False
        assert "strictly positive" in msg

    def test_breadth_evaluates_with_scorecard_isolation(self, monkeypatch):
        from unittest.mock import MagicMock
        from scripts.gate import check_breadth

        # Test that breadth check accurately counts zero vs nonzero and does not bleed
        # Run check_breadth with quick steps=50 on candidate games
        ok, msg, results = check_breadth(game_steps=50, min_nonzero=2)
        # lp85 completes level 1 even in 50-100 steps
        assert isinstance(results, list)
        assert len(results) == 3
        # Verify result structure has game, score, levels, status
        for res in results:
            assert "game" in res
            assert "score" in res
            assert "levels" in res
            assert "status" in res
        # Scorecard isolation: an in-battery entry must reflect a standalone
        # run of that game, not another battery game's scorecard. Re-run the
        # first battery game alone and compare.
        import arc_agi
        from arc_agi import OperationMode

        solo_game = results[0]["game"]
        breadth_res = next(r for r in results if r["game"] == solo_game)
        env_dir = None
        for candidate in (STARTER / "environment_files", WORKSPACE / "environment_files"):
            if candidate.exists():
                env_dir = str(candidate)
                break
        arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL, environments_dir=env_dir)
        card_id = arc.open_scorecard(tags=["gate-isolation-test"])
        env = arc.make(solo_game, scorecard_id=card_id)
        agent_cls = load_agent_class()
        agent = agent_cls(
            card_id=card_id,
            game_id=solo_game,
            agent_name=f"GateIsolation.{solo_game}",
            ROOT_URL="http://localhost",
            record=False,
            arc_env=env,
            budget=50,
        )
        agent.main()
        sc = arc.close_scorecard(card_id)
        solo_score = 0.0
        if sc and sc.environments:
            for env_s in sc.environments:
                if solo_game in env_s.id:
                    solo_score = float(env_s.score)
        assert breadth_res["score"] == round(solo_score, 3), (
            f"{solo_game} breadth entry ({breadth_res['score']}) does not match an "
            f"isolated run ({round(solo_score, 3)}): scorecard bleed detected"
        )

