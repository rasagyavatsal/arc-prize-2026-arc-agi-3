"""Explorer adviser hookup, fallbacks and time guards, with a stand-in model.

No GPU and no real model: ``ScriptedAdviser`` answers from a script behind the
same interface as the vLLM client, and ``HttpAdviser`` is exercised against a
local stub server.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from arcengine import FrameData, GameAction, GameState

from agents import Explorer
from agents.templates import explorer as explorer_mod
from agents.templates.explorer import ADVISER_PROMPTS, Adviser, HttpAdviser, parse_advice


def make_frame(grid, levels=0, avail=(1, 2, 3, 4)):
    return FrameData(
        game_id="test",
        frame=[grid],
        state=GameState.NOT_FINISHED,
        levels_completed=levels,
        win_levels=2,
        guid="guid-1",
        full_reset=False,
        available_actions=list(avail),
    )


class ScriptedAdviser(Adviser):
    """Stand-in model: replays scripted answers and records each prompt."""

    def __init__(self, *answers, delay_s=0.0, error=None, ready=True):
        self.answers = list(answers)
        self.is_ready = ready
        self.prompts = []
        self.delay_s = delay_s
        self.error = error

    def ready(self):
        return self.is_ready

    def advise(self, prompt, timeout_s):
        self.prompts.append(prompt)
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.error:
            raise self.error
        return self.answers.pop(0) if self.answers else None


def make_agent(adviser=None, **kwargs):
    return Explorer(
        card_id="card-1",
        game_id="test",
        agent_name="explorer",
        ROOT_URL="http://localhost:8001",
        record=False,
        arc_env=None,
        adviser=adviser,
        **kwargs,
    )


def board():
    """8x8 board: three single-cell objects of colours 3, 4 and 5."""
    g = [[0] * 8 for _ in range(8)]
    g[1][1] = 3
    g[4][5] = 4
    g[6][2] = 5
    return g


def first_probe(agent, frame):
    action = agent.choose_action([], frame)
    data = action.action_data.model_dump() if action.is_complex() else {}
    return action.value, data.get("x"), data.get("y")


@pytest.mark.unit
class TestParseAdvice:
    def test_extracts_the_json_object(self):
        assert parse_advice('```json\n{"move_first": [3]}\n```') == {"move_first": [3]}
        assert parse_advice('Sure! {"goals": []} hope that helps') == {"goals": []}

    def test_bad_output_is_none(self):
        for text in (None, "", "no json here", "{not json}", "[1, 2]", "{"):
            assert parse_advice(text) is None


@pytest.mark.unit
class TestAdviserHookup:
    def test_no_adviser_keeps_plain_order(self, monkeypatch):
        monkeypatch.delenv("ADVISER_URL", raising=False)
        agent = make_agent()
        assert agent.adviser is None
        assert first_probe(agent, make_frame(board())) == (1, None, None)

    def test_move_advice_reorders_simple_actions(self):
        adviser = ScriptedAdviser({"move_first": ["ACTION4", 3, 9, "junk"]})
        agent = make_agent(adviser)
        assert first_probe(agent, make_frame(board())) == (4, None, None)
        assert agent.advice["actions"] == [4, 3]  # unavailable and junk dropped
        assert [a for a, _, _ in agent._untried_at(agent.last_sig)] == [4, 3, 1, 2]

    def test_click_advice_puts_the_named_object_first(self):
        agent = make_agent(ScriptedAdviser({"click_first": [2], "goals": [0]}))
        frame = make_frame(board(), avail=(6,))
        objects = agent._prompt_objects(board())
        target = dict((o["id"], pt) for o, pt in objects)[2]
        assert first_probe(agent, frame) == (6, *target)
        assert agent.advice["points"][0] == target
        colours = {o["id"]: o["colour"] for o, _ in objects}
        assert agent.advice["colours"] == [colours[2], colours[0]]

    def test_advised_colour_leads_in_other_states(self):
        agent = make_agent(ScriptedAdviser({"click_first": [2]}))
        frame = make_frame(board(), avail=(6,))
        agent.choose_action([], frame)
        colour = agent.advice["colours"][0]
        moved = board()
        for y, row in enumerate(moved):
            for x, v in enumerate(row):
                if v == colour:
                    moved[y][x] = 0
                    moved[7 - y][7 - x] = colour  # same object, elsewhere
        aid, x, y = first_probe(agent, make_frame(moved, avail=(6,)))
        assert moved[y][x] == colour

    def test_advice_never_overrides_known_progress(self):
        agent = make_agent()
        f0 = make_frame(board())
        assert first_probe(agent, f0) == (1, None, None)
        agent.choose_action([], make_frame(board(), levels=1))  # ACTION1 won level 0
        agent.adviser = ScriptedAdviser({"move_first": [4]})
        assert first_probe(agent, f0) == (1, None, None)  # back on level 0
        assert agent.advice["actions"] == [4]

    def test_new_level_drops_the_old_advice_and_asks_again(self):
        adviser = ScriptedAdviser({"move_first": [4]}, {"move_first": [2]})
        agent = make_agent(adviser)
        agent.choose_action([], make_frame(board()))
        assert agent.advice["actions"] == [4]
        assert first_probe(agent, make_frame(board(), levels=1)) == (2, None, None)
        assert len(adviser.prompts) == 2
        assert adviser.prompts[1]["level"] == 2


@pytest.mark.unit
class TestAdviserFallbacks:
    def test_unusable_answer_is_plain_order(self):
        adviser = ScriptedAdviser({"foo": 1})
        agent = make_agent(adviser)
        assert first_probe(agent, make_frame(board())) == (1, None, None)
        assert agent.advice is None
        assert agent.adviser_fails == 1

    def test_unsure_answer_is_plain_order_but_not_a_failure(self):
        adviser = ScriptedAdviser(*[{"player": [], "goals": [], "click_first": [], "move_first": []}] * 5)
        agent = make_agent(adviser)
        for level in range(agent.ADVISER_MAX_FAILS + 2):
            assert first_probe(agent, make_frame(board(), levels=level)) == (1, None, None)
        assert agent.adviser_fails == 0
        assert len(adviser.prompts) == agent.ADVISER_MAX_FAILS + 2
        assert agent.advice is None

    def test_raising_adviser_is_plain_order(self):
        agent = make_agent(ScriptedAdviser(error=RuntimeError("boom")))
        assert first_probe(agent, make_frame(board())) == (1, None, None)
        assert agent.adviser_fails == 1

    def test_repeated_failures_drop_the_adviser(self):
        adviser = ScriptedAdviser()
        agent = make_agent(adviser)
        for level in range(agent.ADVISER_MAX_FAILS + 2):
            agent.choose_action([], make_frame(board(), levels=level))
        assert len(adviser.prompts) == agent.ADVISER_MAX_FAILS

    def test_success_resets_the_failure_count(self):
        adviser = ScriptedAdviser(None, {"move_first": [2]})
        agent = make_agent(adviser)
        agent.choose_action([], make_frame(board()))
        agent.choose_action([], make_frame(board(), levels=1))
        assert agent.adviser_fails == 0


@pytest.mark.unit
class TestAdviserTimeGuards:
    def test_hung_call_is_abandoned_at_the_timeout(self):
        agent = make_agent(ScriptedAdviser({"move_first": [4]}, delay_s=5.0))
        agent.ADVISER_TIMEOUT_S = 0.2
        started = time.time()
        assert first_probe(agent, make_frame(board())) == (1, None, None)
        assert time.time() - started < 2.0
        assert agent.advice is None
        assert 0.2 <= agent.adviser_spent < 2.0

    def test_spent_budget_stops_calls(self):
        adviser = ScriptedAdviser({"move_first": [4]})
        agent = make_agent(adviser)
        agent.adviser_spent = agent.ADVISER_BUDGET_S
        assert first_probe(agent, make_frame(board())) == (1, None, None)
        assert adviser.prompts == []

    def test_timeout_is_capped_by_the_remaining_budget(self):
        adviser = ScriptedAdviser({"move_first": [4]}, delay_s=5.0)
        agent = make_agent(adviser)
        agent.ADVISER_TIMEOUT_S = 30.0
        agent.adviser_spent = agent.ADVISER_BUDGET_S - 0.2
        started = time.time()
        agent.choose_action([], make_frame(board()))
        assert time.time() - started < 2.0

    def test_no_calls_near_the_run_deadline(self, monkeypatch):
        adviser = ScriptedAdviser({"move_first": [4]})
        agent = make_agent(adviser)
        monkeypatch.setattr(
            explorer_mod, "_RUN_START", time.time() - agent.RUN_DEADLINE_S + agent.ADVISER_MARGIN_S - 1
        )
        assert first_probe(agent, make_frame(board())) == (1, None, None)
        assert adviser.prompts == []

    def test_env_configures_timeouts(self, monkeypatch):
        monkeypatch.setenv("ADVISER_TIMEOUT_S", "5")
        monkeypatch.setenv("ADVISER_BUDGET_S", "bad")
        agent = make_agent()
        assert agent.ADVISER_TIMEOUT_S == 5.0
        assert agent.ADVISER_BUDGET_S == Explorer.ADVISER_BUDGET_S


@pytest.mark.unit
class TestAdviserCallPoints:
    def test_plateau_asks_again_up_to_the_cap(self):
        adviser = ScriptedAdviser(*[{"move_first": [1]}] * 5)
        agent = make_agent(adviser)
        agent.ADVISER_PLATEAU = 2
        frame = make_frame([[7]])  # one state: never anything new
        calls = []
        for _ in range(12):
            agent.choose_action([], frame)
            calls.append(len(adviser.prompts))
            agent.action_counter += 1
        assert calls[:6] == [1, 1, 2, 2, 3, 3]
        assert calls[-1] == agent.ADVISER_MAX_CALLS
        assert adviser.prompts[1]["moves_seen"]["ACTION1"]["tries"] == 1

    def test_loading_model_is_waited_for_without_failing(self):
        adviser = ScriptedAdviser({"move_first": [4]}, ready=False)
        agent = make_agent(adviser)
        assert first_probe(agent, make_frame(board())) == (1, None, None)
        assert adviser.prompts == [] and agent.adviser_fails == 0
        adviser.is_ready = True  # the server came up mid-level
        moved = board()
        moved[1][1], moved[2][1] = 0, 3
        assert first_probe(agent, make_frame(moved)) == (4, None, None)
        assert len(adviser.prompts) == 1

    def test_early_call_once_every_move_was_tried(self):
        adviser = ScriptedAdviser(*[{"move_first": [1]}] * 3)
        agent = make_agent(adviser)
        agent.ADVISER_EARLY = True
        g = board()
        calls = []
        for step in range(8):
            g = [row[:] for row in g]
            g[7][step] = 9  # a new state every action: never a plateau
            agent.choose_action([], make_frame(g))
            calls.append(len(adviser.prompts))
            agent.action_counter += 1
        # level start, then once ACTION1-4 have all been tried, then no more
        assert calls == [1, 1, 1, 1, 2, 2, 2, 2]
        assert set(adviser.prompts[1]["moves_seen"]) == {"ACTION1", "ACTION2", "ACTION3", "ACTION4"}

    def test_calls_are_logged(self):
        agent = make_agent(ScriptedAdviser({"move_first": [4]}))
        agent.choose_action([], make_frame(board()))
        assert agent.adviser_log == [{"level": 0, "action": 0, "answer": {"move_first": [4]}}]

    def test_env_knobs(self, monkeypatch):
        monkeypatch.setenv("ADVISER_PLATEAU", "7")
        monkeypatch.setenv("ADVISER_MAX_CALLS", "5")
        monkeypatch.setenv("ADVISER_EARLY", "1")
        agent = make_agent()
        assert (agent.ADVISER_PLATEAU, agent.ADVISER_MAX_CALLS, agent.ADVISER_EARLY) == (7, 5, True)

    def test_no_plateau_call_while_new_states_appear(self):
        adviser = ScriptedAdviser({"move_first": [1]})
        agent = make_agent(adviser)
        agent.ADVISER_EARLY = False  # plateaus only
        for step in range(agent.ADVISER_PLATEAU + 5):
            g = [[0] * 8 for _ in range(8)]
            g[step // 8 % 8][step % 8] = 3
            agent.choose_action([], make_frame(g))
            agent.action_counter += 1
        assert len(adviser.prompts) == 1


@pytest.mark.unit
class TestPrompt:
    def test_prompt_is_a_small_object_list(self):
        adviser = ScriptedAdviser(None, None)
        agent = make_agent(adviser)
        agent.choose_action([], make_frame(board()))  # probes ACTION1
        moved = board()
        moved[1][1], moved[2][1] = 0, 3  # colour 3 moved down a cell
        agent.choose_action([], make_frame(moved))
        prompt = agent._advice_prompt(make_frame(moved), agent._prompt_objects(moved))
        json.dumps(prompt)  # serialisable
        assert prompt["background"] == 0
        assert sorted(o["colour"] for o in prompt["objects"]) == [3, 4, 5]
        assert all(set(o) == {"id", "colour", "size", "box"} for o in prompt["objects"])
        assert prompt["moves_seen"]["ACTION1"] == {"tries": 1, "changed": 1, "colours_changed": [0, 3]}
        assert prompt["actions"] == ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]
        assert "grid" not in prompt

    def test_object_list_is_capped(self):
        agent = make_agent()
        g = [[0] * 16 for _ in range(16)]
        for y in range(0, 16, 2):
            for x in range(0, 16, 2):
                g[y][x] = 1
        objects = agent._prompt_objects(g)
        assert len(objects) == agent.ADVISER_MAX_OBJECTS
        prompt = agent._advice_prompt(make_frame(g), objects)
        assert prompt["objects_not_listed"] > 0


class _Handler(BaseHTTPRequestHandler):
    reply: dict = {}
    status = 200
    delay_s = 0.0
    bodies: list = []

    health = 200

    def do_GET(self):
        self.send_response(type(self).health if self.path == "/health" else 404)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).bodies.append((self.path, body))
        time.sleep(type(self).delay_s)
        payload = json.dumps(type(self).reply).encode()
        self.send_response(type(self).status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


@pytest.fixture
def stub_server():
    handler = type("Handler", (_Handler,), {"bodies": [], "status": 200, "delay_s": 0.0})
    server = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield handler, f"http://127.0.0.1:{server.server_port}/v1"
    server.shutdown()


@pytest.mark.unit
class TestHttpAdviser:
    def test_round_trip(self, stub_server):
        handler, url = stub_server
        handler.reply = {"choices": [{"message": {"content": '{"move_first": [2]}'}}]}
        answer = HttpAdviser(url, model="m").advise({"objects": []}, timeout_s=5)
        assert answer == {"move_first": [2]}
        path, body = handler.bodies[0]
        assert path == "/v1/chat/completions"
        assert body["model"] == "m" and body["reasoning_effort"] == "medium"
        assert body["messages"][0]["content"] == ADVISER_PROMPTS["2"]
        assert json.loads(body["messages"][1]["content"]) == {"objects": []}

    def test_server_error_is_none(self, stub_server):
        handler, url = stub_server
        handler.status = 500
        handler.reply = {"error": "x"}
        assert HttpAdviser(url).advise({}, timeout_s=5) is None

    def test_missing_content_is_none(self, stub_server):
        handler, url = stub_server
        handler.reply = {"choices": [{"message": {"content": None}}]}
        assert HttpAdviser(url).advise({}, timeout_s=5) is None

    def test_slow_server_times_out(self, stub_server):
        handler, url = stub_server
        handler.delay_s = 2.0
        handler.reply = {"choices": [{"message": {"content": "{}"}}]}
        started = time.time()
        assert HttpAdviser(url).advise({}, timeout_s=0.3) is None
        assert time.time() - started < 1.5

    def test_unreachable_server_is_none(self):
        adviser = HttpAdviser("http://127.0.0.1:9/v1")
        assert adviser.ready() is False
        assert adviser.advise({}, timeout_s=1) is None

    def test_readiness_is_polled_then_kept(self, stub_server):
        handler, url = stub_server
        handler.health = 503
        adviser = HttpAdviser(url)
        assert adviser.health_url == url.removesuffix("/v1") + "/health"
        assert adviser.ready() is False
        handler.health = 200
        assert adviser.ready() is False  # not re-checked within the poll interval
        adviser.READY_POLL_S = 0.0
        assert adviser.ready() is True
        handler.health = 503
        assert adviser.ready() is True  # once up, stays up

    def test_env_picks_prompt_and_reasoning(self, monkeypatch, stub_server):
        handler, url = stub_server
        handler.reply = {"choices": [{"message": {"content": "{}"}}]}
        monkeypatch.setenv("ADVISER_URL", url)
        monkeypatch.setenv("ADVISER_PROMPT", "2")
        monkeypatch.setenv("ADVISER_REASONING", "medium")
        make_agent().adviser.advise({}, timeout_s=5)
        _, body = handler.bodies[0]
        assert body["reasoning_effort"] == "medium"
        assert body["messages"][0]["content"] == ADVISER_PROMPTS["2"]

    def test_env_enables_the_http_adviser(self, monkeypatch, stub_server):
        handler, url = stub_server
        handler.reply = {"choices": [{"message": {"content": '{"move_first": [3]}'}}]}
        monkeypatch.setenv("ADVISER_URL", url)
        agent = make_agent()
        assert isinstance(agent.adviser, HttpAdviser)
        assert first_probe(agent, make_frame(board())) == (3, None, None)
