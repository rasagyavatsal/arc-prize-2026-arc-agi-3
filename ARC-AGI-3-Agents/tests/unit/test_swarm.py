"""Unit tests for agents.swarm.Swarm.

The Swarm talks to arc_agi.Arcade; to keep these tests hermetic (no network,
no filesystem scanning) every test injects a FakeArcade by patching
agents.swarm.Arcade before the Swarm is constructed.
"""

from unittest.mock import Mock, patch

import pytest
from arc_agi import OperationMode
from arc_agi.scorecard import (
    EnvironmentScore,
    EnvironmentScoreList,
    EnvironmentScorecard,
)

from agents.swarm import Swarm
from agents.templates.random_agent import Random


class FakeArcade:
    """Hermetic stand-in for arc_agi.Arcade (no network access)."""

    def __init__(self, *args, **kwargs):
        self.operation_mode = OperationMode.OFFLINE
        self.opened_tags = None
        self.closed_ids = []
        self.envs = {}

    def open_scorecard(self, source_url=None, tags=None, opaque=None):
        self.opened_tags = tags
        return "test-card-123"

    def close_scorecard(self, scorecard_id=None):
        self.closed_ids.append(scorecard_id)
        return make_scorecard(scorecard_id or "test-card-123")

    def make(self, game_id, **kwargs):
        env = Mock(name=f"env-{game_id}")
        self.envs[game_id] = env
        return env


def make_scorecard(card_id="test-card-123"):
    """Build a real EnvironmentScorecard offline."""
    run = EnvironmentScore(
        score=10.0,
        levels_completed=1,
        actions=50,
        resets=1,
        completed=True,
    )
    return EnvironmentScorecard(
        card_id=card_id,
        environments=[EnvironmentScoreList(id="game1", runs=[run])],
        tags=["agent", "random"],
    )


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv("ARC_API_KEY", "test-api-key")


def make_swarm(**kwargs):
    """Construct a Swarm with the real Arcade patched out."""
    with patch("agents.swarm.Arcade", FakeArcade):
        defaults = {
            "agent": "random",
            "ROOT_URL": "https://example.com",
            "games": ["game1", "game2"],
        }
        defaults.update(kwargs)
        return Swarm(**defaults)


@pytest.mark.unit
class TestSwarmInitialization:
    def test_swarm_init(self):
        swarm = make_swarm()

        assert swarm.agent_name == "random"
        assert swarm.ROOT_URL == "https://example.com"
        assert swarm.GAMES == ["game1", "game2"]
        assert swarm.agent_class is Random
        assert len(swarm.threads) == 0
        assert len(swarm.agents) == 0
        assert len(swarm.cleanup_threads) == 0

        assert swarm.headers["X-API-Key"] == "test-api-key"
        assert swarm.headers["Accept"] == "application/json"

        # default tags: base tags + agent identity
        assert swarm.tags == ["agent", "random"]

    def test_swarm_arcade_instance(self):
        swarm = make_swarm()
        assert isinstance(swarm._arc, FakeArcade)


@pytest.mark.unit
class TestSwarmScorecard:
    def test_open_scorecard(self):
        swarm = make_swarm()

        card_id = swarm.open_scorecard()
        assert card_id == "test-card-123"
        assert swarm._arc.opened_tags == ["agent", "random"]

    def test_close_scorecard(self):
        swarm = make_swarm()

        scorecard = swarm.close_scorecard("test-card-123")
        assert isinstance(scorecard, EnvironmentScorecard)
        assert scorecard.card_id == "test-card-123"
        assert swarm.card_id is None
        assert swarm._arc.closed_ids == ["test-card-123"]


@pytest.mark.unit
class TestSwarmAgentManagement:
    @patch("agents.swarm.Thread")
    def test_main_orchestration(self, mock_thread):
        mock_thread_instances = [Mock() for _ in range(3)]
        mock_thread.side_effect = mock_thread_instances

        swarm = make_swarm(games=["game1", "game2", "game3"])

        scorecard = swarm.main()

        # one thread per game, started and joined once
        assert mock_thread.call_count == 3
        for mock_thread_instance in mock_thread_instances:
            mock_thread_instance.start.assert_called_once()
            mock_thread_instance.join.assert_called_once()

        # one agent per game, wired to the fake environment
        assert len(swarm.agents) == 3
        assert sorted(swarm._arc.envs) == ["game1", "game2", "game3"]
        for agent, game_id in zip(swarm.agents, ["game1", "game2", "game3"]):
            assert agent.game_id == game_id
            assert agent.card_id == "test-card-123"
            assert agent.agent_name == "random"
            assert agent.tags == ["agent", "random"]
            assert agent.arc_env is swarm._arc.envs[game_id]

        # scorecard lifecycle
        assert swarm._arc.opened_tags == ["agent", "random"]
        assert swarm._arc.closed_ids == ["test-card-123"]
        assert isinstance(scorecard, EnvironmentScorecard)
        assert scorecard.card_id == "test-card-123"
        assert swarm.card_id is None

        # cleanup ran for every agent
        for agent in swarm.agents:
            assert agent._cleanup is False


@pytest.mark.unit
class TestSwarmCleanup:
    def test_cleanup(self):
        swarm = make_swarm()

        mock_agent1 = Mock()
        mock_agent2 = Mock()
        swarm.agents = [mock_agent1, mock_agent2]

        scorecard = make_scorecard()
        swarm.cleanup(scorecard)

        mock_agent1.cleanup.assert_called_once_with(scorecard)
        mock_agent2.cleanup.assert_called_once_with(scorecard)

    def test_cleanup_without_scorecard(self):
        swarm = make_swarm()

        mock_agent = Mock()
        swarm.agents = [mock_agent]

        swarm.cleanup()
        mock_agent.cleanup.assert_called_once_with(None)

    def test_cleanup_closes_session(self):
        swarm = make_swarm()

        mock_agent = Mock()
        swarm.agents = [mock_agent]
        mock_session = Mock()
        swarm._session = mock_session

        swarm.cleanup()
        mock_session.close.assert_called_once()

        # Swarm.__init__ no longer creates a session; cleanup must tolerate that
        delattr(swarm, "_session")
        swarm.cleanup()


@pytest.mark.unit
class TestSwarmTags:
    def test_open_scorecard_with_custom_tags(self):
        """Custom tags are sent first, followed by the base agent tags."""
        custom_tags = ["experiment1", "version2", "test"]

        swarm = make_swarm(games=["game1"], tags=custom_tags)

        assert swarm.tags == custom_tags + ["agent", "random"]

        card_id = swarm.open_scorecard()
        assert card_id == "test-card-123"
        assert swarm._arc.opened_tags == custom_tags + ["agent", "random"]

    def test_open_scorecard_with_empty_tags(self):
        """An explicitly empty tag list still gets the base agent tags."""
        swarm = make_swarm(games=["game1"], tags=[])

        assert swarm.tags == ["agent", "random"]

        card_id = swarm.open_scorecard()
        assert card_id == "test-card-123"
        assert swarm._arc.opened_tags == ["agent", "random"]
