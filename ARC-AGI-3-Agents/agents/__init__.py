from typing import Type, cast

from dotenv import load_dotenv

from .agent import Agent, Playback
from .recorder import Recorder
from .swarm import Swarm
from .templates.random_agent import Random

try:
    from .templates.explorer import Explorer, MyAgent
except ImportError:
    pass

try:
    from .templates.langgraph_functional_agent import LangGraphFunc, LangGraphTextOnly
except ImportError:
    pass

try:
    from .templates.langgraph_random_agent import LangGraphRandom
except ImportError:
    pass

try:
    from .templates.langgraph_thinking import LangGraphThinking
except ImportError:
    pass

try:
    from .templates.llm_agents import LLM, FastLLM, GuidedLLM, ReasoningLLM
except ImportError:
    pass

try:
    from .templates.multimodal import MultiModalLLM
except ImportError:
    pass

try:
    from .templates.reasoning_agent import ReasoningAgent
except ImportError:
    pass

try:
    from .templates.smolagents import SmolCodingAgent, SmolVisionAgent
except ImportError:
    pass

load_dotenv()

AVAILABLE_AGENTS: dict[str, Type[Agent]] = {
    cls.__name__.lower(): cast(Type[Agent], cls)
    for cls in Agent.__subclasses__()
    if cls.__name__ != "Playback"
}

# add all the recording files as valid agent names
for rec in Recorder.list():
    AVAILABLE_AGENTS[rec] = Playback

# update the agent dictionary to include subclasses of LLM class
if "ReasoningAgent" in locals():
    AVAILABLE_AGENTS["reasoningagent"] = ReasoningAgent
if "Explorer" in locals():
    AVAILABLE_AGENTS["myagent"] = Explorer

__all__ = [
    "Swarm",
    "Random",
    "Agent",
    "Recorder",
    "Playback",
    "AVAILABLE_AGENTS",
]
for _symbol in [
    "Explorer",
    "MyAgent",
    "LangGraphFunc",
    "LangGraphTextOnly",
    "LangGraphThinking",
    "LangGraphRandom",
    "LLM",
    "FastLLM",
    "ReasoningLLM",
    "GuidedLLM",
    "ReasoningAgent",
    "SmolCodingAgent",
    "SmolVisionAgent",
    "MultiModalLLM",
]:
    if _symbol in locals():
        __all__.append(_symbol)

