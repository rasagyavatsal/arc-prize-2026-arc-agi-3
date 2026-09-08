# %% [markdown]
"""
# ARC Prize 2026 (ARC-AGI-3) — offline Kaggle submission SKELETON

Run order (Jupyter "percent" format, one `# %%` block per cell):

  CELL 1  discover the competition bundle, set OFFLINE env vars, pip-install
          the bundled wheels with --no-index (never touches the network)
  CELL 2  copy the ARC-AGI-3-Agents framework to the writable working dir,
          make it importable offline, `import agents`
  CELL 3  define + register `MyAgent` (stub policy with TODOs)
  CELL 4  run the Swarm on ONE public game (ls20), print the scorecard JSON
  CELL 5  submission notes (Save Version -> Submit, 9h runtime limit)

Robustness rules baked in:
  * NO internet is ever assumed: pip runs with --no-index --find-links,
    OPERATION_MODE=offline keeps the whole arc_agi stack off the network.
  * The bundle mount name is never hardcoded: /kaggle/input is scanned for a
    directory containing ARC-AGI-3-Agents + arc_agi_3_wheels + environment_files.
  * If /kaggle/input is missing (local smoke test) the script falls back to
    ARC_LOCAL_INPUT (default: /Users/rasagyavatsal/arc-prize-2026-arc-agi-3)
    and writes to ARC_LOCAL_WORKDIR (default: <input>/kaggle_local_run).

This file is a valid python script too: `python submission_skeleton.py`
runs all cells top-to-bottom (that is exactly what Save & Run All does).
"""

# %% ======================================================================
# CELL 1 — discover input, set OFFLINE env vars, install wheels offline
# ========================================================================
import os
import subprocess
import sys
from pathlib import Path

REQUIRED_SUBDIRS = ("ARC-AGI-3-Agents", "arc_agi_3_wheels", "environment_files")


def discover_input_dir() -> Path:
    """Find the competition bundle without assuming its mount name."""
    if os.path.isdir("/kaggle/input"):
        candidates: list[Path] = []
        for name in os.listdir("/kaggle/input"):
            mount = Path("/kaggle/input") / name
            candidates.append(mount)
            if mount.is_dir():  # tolerate one level of nesting
                candidates.extend(p for p in mount.iterdir() if p.is_dir())
        for cand in candidates:
            if all((cand / sub).is_dir() for sub in REQUIRED_SUBDIRS):
                print(f"[skeleton] competition bundle found at: {cand}")
                return cand
        raise FileNotFoundError(
            f"No directory under /kaggle/input contains all of {REQUIRED_SUBDIRS}. "
            f"Seen: {sorted({str(c) for c in candidates})}"
        )
    # Local smoke-test fallback (no Kaggle mount present).
    local = Path(os.environ.get("ARC_LOCAL_INPUT",
                                "/Users/rasagyavatsal/arc-prize-2026-arc-agi-3"))
    print(f"[skeleton] /kaggle/input not found -> local fallback: {local}")
    return local


INPUT_DIR = discover_input_dir()
ON_KAGGLE = str(INPUT_DIR).startswith("/kaggle/input")
WORKING_DIR = (Path("/kaggle/working") if ON_KAGGLE
               else Path(os.environ.get("ARC_LOCAL_WORKDIR",
                                        str(INPUT_DIR / "kaggle_local_run"))))
WORKING_DIR.mkdir(parents=True, exist_ok=True)

WHEELS_DIR = INPUT_DIR / "arc_agi_3_wheels"
ENVIRONMENTS_DIR = INPUT_DIR / "environment_files"
FRAMEWORK_SRC = INPUT_DIR / "ARC-AGI-3-Agents"
PYLIBS = WORKING_DIR / "pylibs"          # writable install target
RECORDINGS_DIR = WORKING_DIR / "recordings"


def set_env_vars() -> None:
    """Full OFFLINE configuration for the arc_agi stack.

    arc_agi.Arcade reads these env vars at construction time (base.py), and
    its load_dotenv() calls never override variables that are already set,
    so anything set here wins over any stray .env file.
    """
    os.environ["OPERATION_MODE"] = "offline"   # local games ONLY, zero network
    os.environ["ENVIRONMENTS_DIR"] = str(ENVIRONMENTS_DIR)
    os.environ["RECORDINGS_DIR"] = str(RECORDINGS_DIR)
    os.environ["ARC_API_KEY"] = ""             # never used in offline mode
    os.environ["ARC_BASE_URL"] = "http://localhost:8001"  # dummy, unused offline
    os.environ["SCHEME"] = "http"              # for Swarm's ROOT_URL cosmetics
    os.environ["HOST"] = "localhost"
    os.environ["PORT"] = "8001"
    os.environ["TESTING"] = "False"            # same default main.py sets
    os.environ.setdefault("DEBUG", "False")


set_env_vars()
os.chdir(WORKING_DIR)  # everything we write lands in the writable dir


def _newest(wheels: list[Path]) -> Path:
    return max(wheels, key=lambda p: [int(x) if x.isdigit() else x
                                      for x in p.name.split("-")[1].split(".")])


# Import names for the runtime deps declared by the two wheels; on Kaggle the
# full closure comes from the bundled wheels, this map is only needed by the
# --no-deps fallback (all of these are preinstalled in the Kaggle image).
_IMPORT_NAMES = {
    "requests": "requests", "flask": "flask", "matplotlib": "matplotlib",
    "pydantic": "pydantic", "python-dotenv": "dotenv", "pillow": "PIL",
    "numpy": "numpy", "python-dateutil": "dateutil",
}


def _wheel_runtime_deps(whl: Path) -> set[str]:
    """Runtime Requires-Dist of a wheel, mapped to import names."""
    import re as _re
    import zipfile
    with zipfile.ZipFile(whl) as zf:
        meta = next(n for n in zf.namelist() if n.endswith(".dist-info/METADATA"))
        text = zf.read(meta).decode("utf-8")
    deps: set[str] = set()
    for line in text.splitlines():
        if line.startswith("Requires-Dist:") and "; extra" not in line:
            name = line.split(":", 1)[1].split(";")[0].strip()
            name = _re.split(r"[<>=!~\[ ]", name)[0]
            deps.add(_IMPORT_NAMES.get(name.lower(), name.lower().replace("-", "_")))
    return deps


def install_wheels_offline() -> Path:
    """pip-install arc_agi + arcengine from the bundled wheels, offline.

    Phase 1 installs the full dependency closure out of the bundled wheels.
    That is the Kaggle path (linux x86_64, py3.12: every bundled wheel
    matches the platform). Phase 2 is a portability fallback for other
    hosts: if the bundled binary wheels do not match, install the two
    pure-python wheels with --no-deps and satisfy their (verified-present)
    dependencies from the preinstalled environment.
    """
    marker = PYLIBS / ".offline_install_ok"
    if marker.exists():
        print(f"[skeleton] wheels already installed in {PYLIBS}")
        return PYLIBS
    arc_agi_wheels = sorted(WHEELS_DIR.glob("arc_agi-*.whl"))
    arcengine_wheels = sorted(WHEELS_DIR.glob("arcengine-*.whl"))
    if not arc_agi_wheels or not arcengine_wheels:
        raise FileNotFoundError(
            f"arc_agi/arcengine wheels not found in {WHEELS_DIR} "
            f"(found {sorted(p.name for p in WHEELS_DIR.glob('*.whl'))})"
        )
    arc_agi_whl, arcengine_whl = _newest(arc_agi_wheels), _newest(arcengine_wheels)
    PYLIBS.mkdir(parents=True, exist_ok=True)

    def _pip(*extra: str) -> list[str]:
        return [
            sys.executable, "-m", "pip", "install",
            "--no-index",                       # NEVER reach out to PyPI
            "--find-links", str(WHEELS_DIR),    # resolve deps from bundled wheels
            "--target", str(PYLIBS),            # input mounts are read-only
            "--no-cache-dir", "--no-compile",
            *extra,
        ]

    cmd = _pip(str(arc_agi_whl), str(arcengine_whl))
    print("[skeleton] RUN:", " ".join(cmd))
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as exc:
        import importlib.util
        needed = _wheel_runtime_deps(arc_agi_whl) | _wheel_runtime_deps(arcengine_whl)
        needed -= {"arc_agi", "arcengine"}  # satisfied by this very install
        missing = sorted(m for m in needed if importlib.util.find_spec(m) is None)
        if missing:
            raise RuntimeError(
                f"offline wheel install failed ({exc}) and these dependencies "
                f"are neither bundled nor preinstalled: {missing}"
            ) from exc
        print("[skeleton] full-closure install failed (bundled binary wheels "
              "likely target another platform); falling back to --no-deps for "
              "the two pure-python wheels")
        subprocess.check_call(_pip("--no-deps", str(arc_agi_whl), str(arcengine_whl)))
    marker.touch()
    return PYLIBS


install_wheels_offline()
if str(PYLIBS) not in sys.path:
    sys.path.insert(0, str(PYLIBS))

import arc_agi   # noqa: E402  (proves the offline install worked)
import arcengine  # noqa: E402
from importlib.metadata import PackageNotFoundError, version  # noqa: E402

try:
    print(f"[skeleton] arc_agi=={version('arc-agi')}  "
          f"arcengine=={version('arcengine')}  (offline import OK)")
except PackageNotFoundError:
    print("[skeleton] arc_agi / arcengine imported (dist metadata not found)")

# %% ======================================================================
# CELL 2 — copy the agent framework into the writable dir, import `agents`
# ========================================================================
# /kaggle/input is READ-ONLY, so the framework is copied to the working dir.
# The offline wheel bundle does NOT include the optional template deps
# (langgraph, langsmith, openai, smolagents), which the stock
# agents/__init__.py imports eagerly. If that import fails, we rewrite the
# __init__ of the WORKING COPY ONLY (repo stays untouched) so the package
# imports with just its core (Agent/Playback/Recorder/Swarm) + whatever
# templates happen to be importable.
import shutil  # noqa: E402

FRAMEWORK_DIR = WORKING_DIR / "ARC-AGI-3-Agents"

if FRAMEWORK_DIR.exists():
    print(f"[skeleton] framework copy already present: {FRAMEWORK_DIR}")
else:
    shutil.copytree(
        FRAMEWORK_SRC,
        FRAMEWORK_DIR,
        ignore=shutil.ignore_patterns(
            ".git", ".github", "__pycache__", "*.pyc", ".venv", "venv",
            ".env", "uv.lock", "logs.log", ".pytest_cache", ".ruff_cache",
            ".mypy_cache", ".pre-commit-config.yaml",
        ),
    )
    print(f"[skeleton] framework copied to {FRAMEWORK_DIR}")

if str(FRAMEWORK_DIR) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK_DIR))

OFFLINE_AGENTS_INIT = '''
"""Offline-safe agents package init (written by the Kaggle skeleton).

Core modules always load; template agents with heavy optional deps
(langgraph/openai/smolagents/langsmith) load on a best-effort basis.
"""
from importlib import import_module
from typing import Type, cast

from dotenv import load_dotenv

load_dotenv()

from .agent import Agent, Playback  # noqa: E402
from .recorder import Recorder  # noqa: E402
from .swarm import Swarm  # noqa: E402

_OPTIONAL_TEMPLATES = (
    ("langgraph_functional_agent", ("LangGraphFunc", "LangGraphTextOnly")),
    ("langgraph_random_agent", ("LangGraphRandom",)),
    ("langgraph_thinking", ("LangGraphThinking",)),
    ("llm_agents", ("LLM", "FastLLM", "GuidedLLM", "ReasoningLLM")),
    ("multimodal", ("MultiModalLLM",)),
    ("random_agent", ("Random",)),
    ("reasoning_agent", ("ReasoningAgent",)),
    ("smolagents", ("SmolCodingAgent", "SmolVisionAgent")),
)

for _mod_name, _names in _OPTIONAL_TEMPLATES:
    try:
        _module = import_module(f".templates.{_mod_name}", __name__)
    except Exception:  # missing optional dep (e.g. offline wheel bundle)
        continue
    for _name in _names:
        globals()[_name] = getattr(_module, _name)

AVAILABLE_AGENTS: dict[str, Type[Agent]] = {
    cls.__name__.lower(): cast(Type[Agent], cls)
    for cls in Agent.__subclasses__()
    if cls.__name__ != "Playback"
}

# recordings double as playable agents (Playback)
try:
    for _rec in Recorder.list():
        AVAILABLE_AGENTS[_rec] = Playback
except Exception:
    pass

__all__ = ["Agent", "Playback", "Recorder", "Swarm", "AVAILABLE_AGENTS"]
'''


def _purge_agents_modules() -> None:
    for name in [m for m in sys.modules if m == "agents" or m.startswith("agents.")]:
        del sys.modules[name]


try:
    import agents  # normal path: works when optional template deps exist
except ImportError as exc:
    print(f"[skeleton] plain 'import agents' failed: {exc!r}")
    print("[skeleton] -> writing offline-safe agents/__init__.py into the WORKING COPY")
    (FRAMEWORK_DIR / "agents" / "__init__.py").write_text(
        OFFLINE_AGENTS_INIT.lstrip("\n"), encoding="utf-8"
    )
    _purge_agents_modules()  # drop the half-initialised package
    import agents

from agents import AVAILABLE_AGENTS, Agent, Swarm  # noqa: E402, F401

print("[skeleton] `agents` imported. AVAILABLE_AGENTS:",
      sorted(AVAILABLE_AGENTS) or "(empty - register your agent in CELL 3)")

# %% ======================================================================
# CELL 3 — define + register the stub agent `MyAgent`
# ========================================================================
import random  # noqa: E402

from arcengine import FrameData, GameAction, GameState  # noqa: E402


class MyAgent(Agent):
    """SKELETON agent: starts the game, then takes persistent random actions.

    Replace the TODO blocks with the real policy. Everything lives on `self`:
      self.frames          full FrameData history (frames[0] = initial)
      self.game_id         e.g. "ls20"
      self.arc_env         EnvironmentWrapper (observation_space = last frame)
      self.action_counter  actions taken so far (hard stop at MAX_ACTIONS)
    and `choose_action` receives the latest frame with:
      .state               NOT_PLAYED / NOT_FINISHED / WIN / GAME_OVER
      .frame               2D grid of colour codes (list[list[int]])
      .available_actions   list[int] of valid GameAction ids right now
      .levels_completed    current score signal
    """

    MAX_ACTIONS = 80  # TODO: raise for real runs (each action = one frame)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rng = random.Random(f"{self.game_id}-{self.card_id}")
        # TODO: persistent per-game memory, e.g.
        #   self.visited_hashes: set[str] = set()      # seen frame signatures
        #   self.action_outcomes: dict[tuple, object]  # action -> observed delta

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Decide if the agent is done playing."""
        # TODO: real termination logic; WIN is the terminal state to reach.
        return latest_frame.state is GameState.WIN

    def choose_action(self, frames, latest_frame) -> GameAction:
        """Pick the next action. THIS IS WHERE THE REAL LOGIC GOES."""
        # ------------------------------------------------------------------
        # TODO 1: build your policy over latest_frame.frame / .state /
        #         .available_actions and self.frames (e.g. detect repeats by
        #         diffing consecutive grids, plan ACTION1..7 sequences, use
        #         ACTION6 clicks on 64x64 coordinates deliberately).
        # ------------------------------------------------------------------

        # (Re)start first: RESET is always free and required after GAME_OVER.
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            return GameAction.RESET

        # Map available action ids -> GameAction members.
        available: list[GameAction] = []
        for aid in (latest_frame.available_actions or []):
            try:
                available.append(GameAction.from_id(aid))
            except ValueError:
                continue
        if not available:
            return GameAction.RESET

        # ------------------------------------------------------------------
        # TODO 2: replace the random pick below with deliberate selection.
        # ------------------------------------------------------------------
        action = self.rng.choice(available)

        # Complex actions need x/y (click position on the 64x64 grid);
        # simple actions just need a free-form reasoning blob.
        if action.is_complex():
            action.set_data({
                "x": self.rng.randint(0, 63),  # TODO: pick a meaningful target
                "y": self.rng.randint(0, 63),
            })
            action.reasoning = {
                "desired_action": action.value,
                "my_reason": "stub random policy",
            }
        else:
            action.reasoning = f"stub simple action {action.name}"
        return action


# The Swarm resolves agents by lowercase class name at run time, so register
# explicitly (AVAILABLE_AGENTS was frozen when `agents` was imported).
AVAILABLE_AGENTS[MyAgent.__name__.lower()] = MyAgent
print(f"[skeleton] registered MyAgent as AVAILABLE_AGENTS['{MyAgent.__name__.lower()}']")

# %% ======================================================================
# CELL 4 — run the Swarm on ONE public game (ls20), print the scorecard
# ========================================================================
import json    # noqa: E402
import logging  # noqa: E402

set_env_vars()  # re-assert (cheap insurance against any .env load ordering)
logging.basicConfig(level=logging.INFO, force=True,
                    format="%(asctime)s | %(levelname)s | %(message)s")

from arc_agi import Arcade  # noqa: E402

# Sanity check: offline game discovery from ENVIRONMENTS_DIR.
arc = Arcade()  # noqa: F841  (scans ENVIRONMENTS_DIR because OPERATION_MODE=offline)
all_games = sorted({e.game_id.split("-")[0] for e in arc.get_environments()})
print("[skeleton] locally available games:", all_games)

# TODO: for the real submission, iterate ALL public games (Swarm fans one
# agent thread out per game) instead of a single one.
GAME_ID = "ls20" if "ls20" in all_games else all_games[0]
AGENT_NAME = MyAgent.__name__.lower()
ROOT_URL = f"http://{os.environ.get('HOST', 'localhost')}:{os.environ.get('PORT', '8001')}"

print(f"[skeleton] Swarm: agent={AGENT_NAME} game={GAME_ID} ROOT_URL={ROOT_URL} (offline)")
swarm = Swarm(AGENT_NAME, ROOT_URL, [GAME_ID], tags=["kaggle", "skeleton", "offline"])
scorecard = swarm.main()

print("=" * 72)
if scorecard is not None:
    print(json.dumps(scorecard.model_dump(mode="json"), indent=2, default=str))
else:
    print("[skeleton] Swarm returned no scorecard - inspect the log above.")
print("=" * 72)
# Artifacts to inspect in /kaggle/working:
#   recordings/*.recording.jsonl   full action/frame replay of the run
print("[skeleton] recordings:", sorted(p.name for p in RECORDINGS_DIR.glob('*')) if RECORDINGS_DIR.exists() else "none")

# %% [markdown]
# ## CELL 5 — How to submit
#
# 1. **Attach data**: in the Kaggle editor add the competition bundle
#    (`arc-prize-2026-arc-agi-3`) via *Add Input*; CELL 1 discovers the mount
#    dynamically, so the exact dataset name does not matter.
# 2. **Internet OFF** (it is forced for this competition anyway): CELL 1 uses
#    `pip --no-index --find-links` on the bundled wheels and
#    `OPERATION_MODE=offline` keeps the whole stack local. Any accidental
#    network call is a bug — the notebook must never depend on one.
# 3. **Runtime**: CPU or GPU session, hard limit **9 hours**. Keep
#    *Save Version -> Save & Run All (Commit)* under that budget; the run must
#    finish cleanly (a crashed notebook cannot be submitted).
# 4. **Submit**: after the committed run finishes, click *Submit* on the
#    version (or *Submit to Competition* from the version's Options menu) and
#    pick `arc-prize-2026-arc-agi-3`.
# 5. **Scale up before submitting**: replace the stub in CELL 3 with your real
#    policy, raise `MAX_ACTIONS`, and run the Swarm over all 25 public games
#    (one agent thread per game) instead of the single `ls20` smoke run.
#    Recordings and the scorecard JSON in `/kaggle/working` are your debugging
#    artifacts (they are not part of the submission itself).
