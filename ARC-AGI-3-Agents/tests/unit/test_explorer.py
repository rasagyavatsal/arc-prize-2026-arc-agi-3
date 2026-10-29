"""Fast, pure-python unit tests for the Explorer agent.

These tests stub FrameData directly and never run a real game or network call.
"""

import pytest
from arcengine import FrameData, GameAction, GameState

from agents import Explorer


def make_frame(
    grid,
    state=GameState.NOT_FINISHED,
    levels=0,
    avail=(1, 2, 3, 4),
    full_reset=False,
):
    """Build a minimal FrameData stub (single settled layer)."""
    return FrameData(
        game_id="test",
        frame=[grid],
        state=state,
        levels_completed=levels,
        win_levels=1,
        guid="guid-1",
        full_reset=full_reset,
        available_actions=list(avail),
    )


def make_agent() -> Explorer:
    return Explorer(
        card_id="card-1",
        game_id="test",
        agent_name="explorer",
        ROOT_URL="http://localhost:8001",
        record=False,
        arc_env=None,
    )


@pytest.mark.unit
class TestExplorer:
    def test_resets_on_not_played_and_game_over_then_gives_up(self):
        agent = make_agent()

        # First ever RESET (NOT_PLAYED, no actions taken yet) is free.
        assert agent.choose_action([], make_frame([[0]], state=GameState.NOT_PLAYED)) is GameAction.RESET
        assert agent.level_attempts == {}

        # Each GAME_OVER reset counts as a level attempt.
        assert agent.choose_action([], make_frame([[0]], state=GameState.GAME_OVER)) is GameAction.RESET
        assert agent.level_attempts[0] == 1
        assert agent.choose_action([], make_frame([[0]], state=GameState.GAME_OVER)) is GameAction.RESET
        assert agent.level_attempts[0] == 2

        # Once the attempt cap is reached and nothing is left to explore,
        # the agent gives up and is_done becomes True.
        for _ in range(2 * agent.MAX_LEVEL_ATTEMPTS):
            agent.choose_action([], make_frame([[0]], state=GameState.GAME_OVER))
            if agent._give_up:
                break
        assert agent._give_up is True
        assert agent.level_attempts[0] == agent.MAX_LEVEL_ATTEMPTS
        assert agent.is_done([], make_frame([[0]])) is True

    def test_memory_records_action_effects_and_level_delta(self):
        agent = make_agent()
        f0 = make_frame([[0, 0], [0, 0], [0, 0], [0, 0]])
        sig0 = agent._signature(f0)

        # Fresh state: first untried available action is probed (ACTION1).
        action = agent.choose_action([], f0)
        assert action is GameAction.ACTION1
        assert agent.pending is not None and agent.pending["key"] == "ACTION1"

        # Result frame: grid changed -> edge recorded with the result signature.
        f1 = make_frame([[1, 0], [0, 0], [0, 0], [0, 0]])
        agent.choose_action([], f1)
        edge = agent.memory[sig0]["edges"]["ACTION1"]
        sig1 = agent._signature(f1)
        assert edge["changed"] is True
        assert edge["dlevel"] == 0
        assert edge["result"] == sig1
        assert edge["tries"] == 1
        # a new probe from the new state is already queued
        assert agent.pending is not None and agent.pending["sig"] == sig1

        # A result frame with levels_completed increased records the delta.
        f2 = make_frame([[1, 0], [0, 0], [0, 0], [0, 0]], levels=1)
        agent.choose_action([], f2)
        edge = agent.memory[sig1]["edges"]["ACTION1"]
        assert edge["dlevel"] == 1
        assert agent.last_levels == 1

    def test_action6_probes_use_available_actions_and_colour_centroids(self):
        agent = make_agent()
        grid = [[0] * 8 for _ in range(8)]
        for x in range(2, 5):
            for y in range(2, 5):
                grid[y][x] = 9  # 3x3 block, centroid (3, 3)
        grid[6][6] = 5  # rare single pixel, centroid (6, 6)

        # Only ACTION6 is available -> the agent must never emit simple actions.
        f = make_frame(grid, avail=(6,))
        action = agent.choose_action([], f)
        assert action is GameAction.ACTION6
        # Rarest colour is probed first.
        assert (action.action_data.x, action.action_data.y) == (6, 6)

        # Each call records the previous probe's result; next candidate follows.
        sig = agent._signature(f)
        assert "ACTION6@6,6" not in agent.memory[sig]["edges"]
        second = agent.choose_action([], f)
        assert "ACTION6@6,6" in agent.memory[sig]["edges"]
        assert (second.action_data.x, second.action_data.y) == (3, 3)
        # after the colour centroids, the deterministic fallback points follow
        third = agent.choose_action([], f)
        assert (third.action_data.x, third.action_data.y) == (4, 4)

    def test_wedged_state_resets_then_gives_up_and_lru_elsewhere(self):
        # A wedged state (every known action a no-op) gets a sparing RESET
        # immediately, and the agent gives up once attempts run out.
        agent = make_agent()
        f = make_frame([[7]])
        sig = agent._signature(f)
        node = agent._node(sig, f)
        for aid in (1, 2, 3, 4):
            node["edges"][f"ACTION{aid}"] = {
                "tries": 1,
                "result": sig,
                "dlevel": 0,
                "state": GameState.NOT_FINISHED.name,
                "changed": False,
                "full_reset": False,
            }

        assert agent.choose_action([], f) is GameAction.RESET
        assert agent.level_attempts[0] == 1
        for _ in range(2 * agent.MAX_LEVEL_ATTEMPTS):
            agent.choose_action([], f)
            if agent._give_up:
                break
        assert agent._give_up is True
        assert agent.level_attempts[0] == agent.MAX_LEVEL_ATTEMPTS
        assert agent.is_done([], f) is True

        # A state with at least one known effective action is not wedged. Once
        # the LRU retries are exhausted (no frontier reachable anywhere in the
        # memory graph), the cycle is escaped with a bounded random walk
        # (explore mode), which falls back to a RESET once its fruitless runs
        # are used up.  No result frames are fed back here (pending is
        # cleared), so the hand-built edges stay frozen for the whole loop.
        agent = make_agent()
        f = make_frame([[7]])
        sig = agent._signature(f)
        node = agent._node(sig, f)
        for aid, changed in ((1, False), (2, True), (3, False), (4, False)):
            node["edges"][f"ACTION{aid}"] = {
                "tries": 1, "result": sig, "dlevel": 0,
                "state": GameState.NOT_FINISHED.name, "changed": changed,
                "full_reset": False,
            }
        first = agent.choose_action([], f)
        assert first is GameAction.ACTION1  # least-recently-tried, ties by id
        agent.pending = None

        node["stuck"] = agent.MAX_STUCK_LRU  # LRU retries already exhausted
        saw_explore = 0
        action = None
        for _ in range(2 * agent.MAX_EXPLORE_RUNS * agent.EXPLORE_STEPS + 10):
            action = agent.choose_action([], f)
            agent.pending = None
            why = action.reasoning or {}
            if isinstance(why, str):
                why = {"why": why}
            if "explore" in why.get("why", ""):
                saw_explore += 1
            if action is GameAction.RESET:
                break
        assert saw_explore > 0, "expected explore-walk actions before the RESET"
        assert action is GameAction.RESET
        assert agent.level_attempts[0] == 1

    def test_budget_and_time_limit_configuration(self):
        import os
        from agents.templates.explorer import Explorer, MyAgent
        assert MyAgent is Explorer

        # Default budget and time limit
        agent_default = Explorer(
            card_id="c", game_id="g", agent_name="a", ROOT_URL="u", record=False, arc_env=None
        )
        assert agent_default.MAX_ACTIONS == 1500
        assert agent_default.TIME_LIMIT_S == 27000.0

        # Explicit kwargs override
        agent_custom = Explorer(
            card_id="c", game_id="g", agent_name="a", ROOT_URL="u", record=False, arc_env=None,
            budget=500, time_limit_s=120.0
        )
        assert agent_custom.MAX_ACTIONS == 500
        assert agent_custom.TIME_LIMIT_S == 120.0

        # Env var override
        os.environ["MAX_ACTIONS"] = "2000"
        os.environ["TIME_LIMIT_S"] = "250.0"
        try:
            agent_env = Explorer(
                card_id="c", game_id="g", agent_name="a", ROOT_URL="u", record=False, arc_env=None
            )
            assert agent_env.MAX_ACTIONS == 2000
            assert agent_env.TIME_LIMIT_S == 250.0
        finally:
            del os.environ["MAX_ACTIONS"]
            del os.environ["TIME_LIMIT_S"]

    def test_wall_clock_timeout_stops_agent(self):
        import time
        agent = make_agent()
        agent.TIME_LIMIT_S = 0.05
        f = make_frame([[0]])
        # Before timer is set, not done
        assert agent.is_done([], f) is False

        # Set timer in the past
        agent.timer = time.time() - 0.1
        assert agent.is_done([], f) is True

    def test_run_deadline_stops_agent(self):
        agent = make_agent()
        f = make_frame([[0]])
        assert agent.is_done([], f) is False

        # The run-wide deadline applies even when the per-game timer is fresh
        agent.RUN_DEADLINE_S = 0.0
        assert agent.is_done([], f) is True

    def test_autonomy_without_baseline_actions(self):
        """Verify agent never accesses baseline_actions or metadata and adapts total_levels from frame."""
        agent = make_agent()
        # arc_env is None, no baseline_actions exist
        f = make_frame([[1]], levels=2)
        f.win_levels = 10
        agent.choose_action([], f)
        assert agent.total_levels == 10

    def test_status_bar_is_masked_out_of_state_identity(self):
        """A top-row step counter must not make every revisit a new state."""
        agent = make_agent()

        def grid(step, avatar_x):
            g = [[0] * 32 for _ in range(32)]
            g[0][step % 32] = 5  # counter ticks on every action
            g[16][avatar_x] = 3
            return g

        sigs = []
        for step in range(2 * agent.HUD_MIN_OBS):
            agent.choose_action([], make_frame(grid(step, 8 + step % 2)))
            sigs.append(agent.last_sig)

        assert agent.hud_rows == {0}
        # once masked, the avatar's two positions are the only two states
        assert len(set(sigs[-agent.HUD_MIN_OBS:])) == 2
        assert agent.world_changed  # the avatar still moved

    def test_each_reset_attempt_gets_fresh_walks(self):
        """A RESET must not leave the walk budget spent, or the next stuck
        start state answers it with another RESET (a reset loop)."""
        agent = make_agent()
        f = make_frame([[0]])
        agent.explore_runs = agent.MAX_EXPLORE_RUNS
        agent.explore_left = 3
        assert agent._reset_or_giveup(f, "test", sig=agent._signature(f)) is GameAction.RESET
        assert agent.explore_runs == 0 and agent.explore_left == 0

    def test_playfield_lines_are_never_masked(self):
        agent = make_agent()
        for step in range(2 * agent.HUD_MIN_OBS):
            g = [[0] * 32 for _ in range(32)]
            g[16][step % 32] = 3  # mid-field object changes every step
            agent.choose_action([], make_frame(g))
        assert agent.hud_rows == set() and agent.hud_cols == set()

    def test_clicks_on_dead_colours_are_probed_last(self):
        agent = make_agent()
        g = [[0] * 16 for _ in range(16)]
        g[4][4] = 1  # colour 1: clicks have never reacted
        g[10][10] = 2  # colour 2: unknown
        agent.click_stats[1] = [agent.CLICK_DEAD_AFTER, 0]
        agent.choose_action([], make_frame(g, avail=(6,)))
        sig = agent.last_sig

        live = {(x, y) for _, x, y in agent._untried_at(sig)}
        everything = {(x, y) for _, x, y in agent._untried_at(sig, dead_ok=True)}
        assert (10, 10) in live and (4, 4) not in live
        assert (4, 4) in everything
        assert all(agent.memory[sig]["colours"][p] != 1 for p in live)

    def test_state_reached_by_a_click_gets_its_own_probes(self):
        agent = make_agent()
        g0 = [[0] * 16 for _ in range(16)]
        g0[3][3] = 9
        first = agent.choose_action([], make_frame(g0, avail=(6,)))
        assert (first.action_data.x, first.action_data.y) == (3, 3)

        g1 = [[0] * 16 for _ in range(16)]
        g1[12][12] = 9  # the click moved the object across the board
        agent.choose_action([], make_frame(g1, avail=(6,)))
        cands = agent.memory[agent.last_sig]["cands"]
        assert cands[0] in {(3 + dx, 3 + dy) for dx in (-2, -1, 1, 2) for dy in (-2, -1, 1, 2)}
        assert (12, 12) in cands  # not only the cells around the click

    def test_inert_click_targets_are_deferred_in_other_states(self):
        agent = make_agent()

        def board(ox, oy):
            g = [[0] * 16 for _ in range(16)]
            for dx, dy in ((0, 0), (1, 0), (0, 1)):  # an L-shaped object
                g[oy + dy][ox + dx] = 4
            return g

        f = make_frame(board(2, 2), avail=(6,))
        action = agent.choose_action([], f)
        point = (action.action_data.x, action.action_data.y)
        agent.choose_action([], f)  # same frame back: the click did nothing
        target = agent._click_targets(board(2, 2), [point])[point]
        assert target[0] == "obj"  # the first probe lands on the object
        assert agent.target_stats[target] == [1, 0]

        # the same object somewhere else, clicked on the same cell of its shape
        moved = board(8, 5)
        agent.pending = None
        agent.choose_action([], make_frame(moved, avail=(6,)))
        sig = agent.last_sig
        shifted = (point[0] + 6, point[1] + 3)
        assert agent.memory[sig]["targets"][shifted] == target
        assert shifted not in {(x, y) for _, x, y in agent._untried_at(sig)}
        assert shifted in {(x, y) for _, x, y in agent._untried_at(sig, dead_ok=True)}

    def test_life_lost_back_at_the_start_keeps_the_real_edge(self):
        """A step counter that runs out sends the player back to the start;
        that must not overwrite where the move really goes."""
        agent = make_agent()

        def board(x):
            g = [[0] * 8 for _ in range(8)]
            g[2][x] = 3
            return g

        agent.choose_action([], make_frame(board(1)))  # start; probes ACTION1
        start = agent.last_sig
        agent.choose_action([], make_frame(board(2)))  # moved; probes ACTION1
        mid = agent.last_sig
        agent.choose_action([], make_frame(board(3)))
        far = agent.last_sig
        assert agent.memory[mid]["edges"]["ACTION1"]["result"] == far

        # the same move again, but the life ran out on it
        agent.pending = {"sig": mid, "key": "ACTION1", "levels": 0, "seen": set()}
        agent.choose_action([], make_frame(board(1)))
        edge = agent.memory[mid]["edges"]["ACTION1"]
        assert agent.last_sig == start
        assert edge["result"] == far and edge["lost"] == 1 and edge["tries"] == 2
