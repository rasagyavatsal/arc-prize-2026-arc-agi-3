"""Explorer: a deterministic, no-LLM exploration baseline for ARC-AGI-3.

State identity
--------------
Games composite HUD overlays (step-budget bars, lives, tickers) onto the world
frame, and many levels animate per move, so a raw grid hash makes every revisit
of the same world cell look like a new state.  Explorer therefore identifies a
state by the position of the *movers*: the connected clusters of cells that
changed on the last transition whose new value differs from the frame's
modal (background) colour.  A movement of the avatar produces one such cluster,
a pushed box a second one, and blocked moves or HUD ticks produce none (the
previous mover position is kept).  States without any observed mover fall back
to a full grid hash.

Memory
------
A persistent dict (kept across levels) maps ``state signature -> {action key:
observed delta}`` where a delta records the resulting signature, whether the
state changed, the levels_completed delta and the resulting game state.

Per step the agent:
1. RESETs on NOT_PLAYED / GAME_OVER / empty frames (sparing, budgeted).
2. Takes a known levels-increasing action when one exists.
3. Probes untried available actions; ACTION6 probes centroids of distinct
   coloured regions (rare colours first) instead of random coordinates, and
   ACTION7 (frequently an undo/state-restore) is probed before ACTION6.
4. Otherwise replays known paths (BFS over the memory graph) to the nearest
   state that still has untried actions.
5. Avoids no-op loops: cycles fall back to the least-recently-tried action,
   then a sparing level RESET, then the agent stops honestly.

The agent uses no LLM and no network calls; it runs on CPU.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from collections import Counter, deque
from typing import Any, Optional

from arcengine import FrameData, GameAction, GameState

try:
    from agents.agent import Agent
except ImportError:
    try:
        from ..agent import Agent
    except (ImportError, ValueError):
        import sys
        from pathlib import Path
        cur = Path(__file__).resolve()
        for parent in cur.parents:
            if (parent / "agents" / "agent.py").exists():
                if str(parent) not in sys.path:
                    sys.path.insert(0, str(parent))
                break
            if (parent / "vendor" / "ARC-AGI-3-Agents" / "agents" / "agent.py").exists():
                vpath = str(parent / "vendor" / "ARC-AGI-3-Agents")
                if vpath not in sys.path:
                    sys.path.insert(0, vpath)
                break
        from agents.agent import Agent

logger = logging.getLogger(__name__)


class Explorer(Agent):
    """Persistent-memory frontier explorer (programmatic, CPU-only)."""

    DEFAULT_MAX_ACTIONS = 1500
    DEFAULT_TIME_LIMIT_S = 270.0  # < 4.9 min / 294s per game (scalable to 110 games in 9h)
    MAX_ACTIONS = DEFAULT_MAX_ACTIONS
    TIME_LIMIT_S = DEFAULT_TIME_LIMIT_S
    RESERVE_ACTIONS = 2  # actions kept for a final known-progress push
    RECENT_WINDOW = 16  # signature history length for cycle detection
    NO_CHANGE_LIMIT = 6  # no-effect actions tolerated before a level RESET
    MAX_LEVEL_ATTEMPTS = 6  # resets/game-overs per level before giving up
    MAX_STUCK_LRU = 3  # least-recently-tried retries before a level RESET
    MAX_PROBE_COORDS = 12  # ACTION6 coordinates probed per state signature
    MOVER_MIN_CELLS = 8  # smallest changed cluster treated as a mover
    DEMOTE_AFTER = 3  # consecutive no-ops before an action is probed last
    HUD_BOTTOM_ROWS = 2  # bottom rows ignored (status bars/timers live there)

    def __init__(
        self,
        *args: Any,
        budget: int | None = None,
        time_limit_s: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if budget is not None:
            self.MAX_ACTIONS = int(budget)
        elif "MAX_ACTIONS" in os.environ:
            try:
                self.MAX_ACTIONS = int(os.environ["MAX_ACTIONS"])
            except ValueError:
                self.MAX_ACTIONS = self.DEFAULT_MAX_ACTIONS
        else:
            self.MAX_ACTIONS = self.DEFAULT_MAX_ACTIONS

        if time_limit_s is not None:
            self.TIME_LIMIT_S = float(time_limit_s)
        elif "TIME_LIMIT_S" in os.environ:
            try:
                self.TIME_LIMIT_S = float(os.environ["TIME_LIMIT_S"])
            except ValueError:
                self.TIME_LIMIT_S = self.DEFAULT_TIME_LIMIT_S
        else:
            self.TIME_LIMIT_S = self.DEFAULT_TIME_LIMIT_S

        self.timer: Optional[float] = None
        # signature -> {"avail": [ids], "edges": {key: delta}, "visits": int,
        #               "cands": [(x, y)], "stuck": int}
        self.memory: dict[str, dict[str, Any]] = {}
        self.pending: Optional[dict[str, Any]] = None
        self.plan: list[dict[str, Any]] = []
        self.recent: deque[str] = deque(maxlen=self.RECENT_WINDOW)
        self._seen: set[str] = set()
        self.no_change_streak = 0
        self.level_attempts: dict[int, int] = {}
        self.last_levels = -1
        self.last_sig = ""
        self.level_start_action = 0
        # mover tracking (see module docstring)
        self.last_grid: Optional[list[list[int]]] = None
        self.last_grid_levels = -1
        self.mover_centroids: tuple[tuple[int, int], ...] = ()
        self._reset_pending = False  # last issued action was RESET
        self.action_nop_streak: dict[int, int] = {}
        self._give_up = False
        self.total_levels = 8  # default baseline; updated live from FrameData.win_levels
        logger.info(
            f"{self.game_id}: explorer ready "
            f"(levels={self.total_levels or '?'}, budget={self.MAX_ACTIONS}, time_limit={self.TIME_LIMIT_S}s)"
        )

    # ------------------------------------------------------------------ memory

    def _update_total_levels(self, frame: FrameData) -> None:
        """Derive total levels strictly from observation FrameData (win_levels)."""
        win_levels = getattr(frame, "win_levels", None)
        if win_levels is not None:
            try:
                val = int(win_levels)
                if val > 0:
                    self.total_levels = val
            except (ValueError, TypeError):
                pass

    @staticmethod
    def _grid(frame: FrameData) -> list[list[int]]:
        """The settled view of the world is the last rendered layer."""
        return frame.frame[-1] if frame.frame else []

    def _observe_transition(self, grid: list[list[int]], levels: int) -> None:
        """Track mover positions from the last transition.

        A mover is a connected cluster of >= MOVER_MIN_CELLS changed cells whose
        new value differs from the frame's modal colour (HUD ticks and floor
        reveal are modal-coloured and thus ignored).  When no mover moved, the
        previous mover position is kept, so blocked moves and HUD-only changes
        preserve state identity.
        """
        was_reset = self._reset_pending
        self._reset_pending = False
        same_level = (
            self.last_grid is not None
            and levels == self.last_grid_levels
            and len(self.last_grid) == len(grid)
        )
        if not same_level or was_reset:
            # a RESET (or a level change) restores the world and the HUD bars,
            # which would otherwise show up as fake movers
            self.mover_centroids = ()
        else:
            # status bars/timers are conventionally drawn along the bottom
            # edge; changes confined to that band are HUD ticks, not world
            # changes, and must not make no-ops look effective
            cutoff = max(0, min(len(self.last_grid), len(grid)) - self.HUD_BOTTOM_ROWS)
            changed = [
                (y, x)
                for y in range(cutoff)
                for x, (va, vb) in enumerate(zip(self.last_grid[y], grid[y]))
                if va != vb
            ]
            cents: list[tuple[int, int]] = []
            if changed:
                modal = Counter(
                    v for row in grid[:cutoff] for v in row
                ).most_common(1)[0][0]
                remaining = set(changed)
                for start in changed:
                    if start not in remaining:
                        continue
                    # flood fill one changed cluster (4-neighbourhood)
                    stack = [start]
                    remaining.discard(start)
                    cluster: list[tuple[int, int]] = []
                    while stack:
                        y, x = stack.pop()
                        cluster.append((y, x))
                        for nb in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                            if nb in remaining:
                                remaining.discard(nb)
                                stack.append(nb)
                    if len(cluster) < self.MOVER_MIN_CELLS:
                        continue
                    arrival = [(y, x) for y, x in cluster if grid[y][x] != modal]
                    pts = arrival or cluster
                    cy = sum(p[0] for p in pts) // len(pts)
                    cx = sum(p[1] for p in pts) // len(pts)
                    cents.append((cy, cx))
            if cents:
                self.mover_centroids = tuple(sorted(cents))
        self.last_grid = [row[:] for row in grid]
        self.last_grid_levels = levels

    def _signature(self, frame: FrameData) -> str:
        """State signature: mover positions when known, else the grid hash."""
        levels = int(frame.levels_completed)
        if self.mover_centroids:
            pos = ";".join(f"{y}.{x}" for y, x in self.mover_centroids)
            return f"mv|{levels}|{pos}"
        grid = self._grid(frame)
        cutoff = max(0, len(grid) - self.HUD_BOTTOM_ROWS)
        h = hashlib.sha1()
        for row in grid[:cutoff]:
            h.update((";" + ",".join(str(int(v)) for v in row)).encode())
        return f"gd|{levels}|{h.hexdigest()[:16]}"

    def _node(self, sig: str, frame: FrameData) -> dict[str, Any]:
        node = self.memory.get(sig)
        cur_avail = {int(a) for a in (frame.available_actions or [])}
        if node is None:
            node = {
                "avail": sorted(a for a in cur_avail if 0 <= a <= 7),
                "edges": {},
                "visits": 0,
                "cands": self._candidates(frame),
                "stuck": 0,
            }
            self.memory[sig] = node
        else:
            # available_actions can be state dependent; keep the union.
            node["avail"] = sorted(
                set(node["avail"]) | {a for a in cur_avail if 0 <= a <= 7}
            )
        return node

    def _candidates(self, frame: FrameData) -> list[tuple[int, int]]:
        """ACTION6 probe coordinates: centroids of distinct colours, rare first."""
        grid = self._grid(frame)
        height = len(grid)
        width = len(grid[0]) if grid else 0
        if not width or not height:
            return [(32, 32)]
        cells: dict[int, list[tuple[int, int]]] = {}
        for y, row in enumerate(grid):
            for x, value in enumerate(row):
                if value:
                    cells.setdefault(int(value), []).append((x, y))
        cands: list[tuple[int, int]] = []
        for color in sorted(cells, key=lambda c: (len(cells[c]), c)):
            pts = cells[color]
            cx = min(63, max(0, sum(p[0] for p in pts) // len(pts)))
            cy = min(63, max(0, sum(p[1] for p in pts) // len(pts)))
            if (cx, cy) not in cands:
                cands.append((cx, cy))
            if len(cands) >= self.MAX_PROBE_COORDS:
                return cands
        for x, y in (
            (width // 2, height // 2),
            (width // 4, height // 4),
            (3 * width // 4, height // 4),
            (width // 4, 3 * height // 4),
            (3 * width // 4, 3 * height // 4),
        ):
            if len(cands) >= 4:
                break
            if (x, y) not in cands:
                cands.append((x, y))
        return cands[: self.MAX_PROBE_COORDS]

    @staticmethod
    def _key(aid: int, x: Optional[int], y: Optional[int]) -> str:
        name = GameAction.from_id(int(aid)).name
        return name if x is None else f"{name}@{int(x)},{int(y)}"

    @staticmethod
    def _parse_key(key: str) -> tuple[int, Optional[int], Optional[int]]:
        if "@" in key:
            name, coords = key.split("@", 1)
            x_s, y_s = coords.split(",", 1)
            return GameAction.from_name(name).value, int(x_s), int(y_s)
        return GameAction.from_name(key).value, None, None

    @staticmethod
    def _key_action_id(key: str) -> Optional[int]:
        """Action id of a simple-action edge key (None for coords/RESET)."""
        if "@" in key or key.startswith("RESET"):
            return None
        try:
            return GameAction.from_name(key).value
        except ValueError:
            return None

    def _demoted(self, aid: int) -> bool:
        return self.action_nop_streak.get(aid, 0) >= self.DEMOTE_AFTER

    @staticmethod
    def _to_action(aid: int) -> Optional[GameAction]:
        try:
            return GameAction.from_id(int(aid))
        except ValueError:
            return None

    # ------------------------------------------------------------- memory query

    def _untried_at(self, sig: str) -> list[tuple[int, Optional[int], Optional[int]]]:
        """Untried (action, x, y) probes at a state, best first.

        Simple actions first in id order (ACTION7 before ACTION6, since ACTION7
        is frequently a cheap undo), then ACTION6 probe coordinates.
        """
        node = self.memory.get(sig)
        if not node:
            return []
        out: list[tuple[int, Optional[int], Optional[int]]] = []
        deferred: list[tuple[int, Optional[int], Optional[int]]] = []
        for aid in sorted(node["avail"]):
            act = self._to_action(aid)
            if act is None or act is GameAction.RESET or act.is_complex():
                continue
            if act.name not in node["edges"]:
                # actions that keep doing nothing are probed last, so a
                # gravity-style game does not waste two probes per node
                (deferred if self._demoted(aid) else out).append((aid, None, None))
        out.extend(deferred)
        for aid in sorted(node["avail"]):
            act = self._to_action(aid)
            if act is None or not act.is_complex():
                continue
            for x, y in node["cands"]:
                if self._key(aid, x, y) not in node["edges"]:
                    out.append((aid, x, y))
        return out

    def _has_untried(self, sig: str) -> bool:
        return bool(self._untried_at(sig))

    def _progress_action(
        self, sig: str
    ) -> Optional[tuple[int, Optional[int], Optional[int]]]:
        """A known action from this state that increased levels_completed."""
        node = self.memory.get(sig)
        if not node:
            return None
        for key, edge in node["edges"].items():
            if key.startswith("RESET"):
                continue
            if edge.get("dlevel", 0) > 0 and edge.get("result") and edge["result"] != sig:
                return self._parse_key(key)
        return None

    def _lru_action(
        self, sig: str
    ) -> Optional[tuple[int, Optional[int], Optional[int]]]:
        """Least-recently-tried allowed action (cycle breaker)."""
        node = self.memory.get(sig)
        if not node:
            return None
        best: Optional[tuple[int, Optional[int], Optional[int]]] = None
        best_rank: Optional[tuple[int, int, int, int]] = None
        for aid in sorted(node["avail"]):
            act = self._to_action(aid)
            if act is None or act is GameAction.RESET:
                continue
            if act.is_complex():
                options: list[tuple[Optional[int], Optional[int]]] = list(node["cands"])
            else:
                options = [(None, None)]
            for x, y in options:
                edge = node["edges"].get(self._key(aid, x, y))
                tries = int(edge.get("tries", 0)) if edge else 0
                # prefer edges labelled undo (e.g. ACTION7 restore) on ties
                undo_rank = 0 if (edge and edge.get("undo")) else 1
                rank = (tries, undo_rank, aid, -1 if x is None else int(x))
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best = (aid, x, y)
        return best

    def _level_has_frontier(self, sig: Optional[str], latest: FrameData) -> bool:
        """True when the level's memory graph still offers something untried."""
        if sig is None:
            return False
        if self._untried_at(sig) or self._has_reachable_frontier(sig):
            return True
        # the reset lands on the level's start state; a frontier known from any
        # previously seen state of this level counts as remaining work
        levels = int(latest.levels_completed)
        for other in self._seen:
            if f"|{levels}|" in other and self._has_untried(other):
                return True
        return False

    def _has_reachable_frontier(self, sig: str) -> bool:
        node = self.memory.get(sig)
        if not node:
            return False
        parent: dict[str, str] = {}
        seen = {sig}
        queue: deque[str] = deque([sig])
        while queue:
            cur = queue.popleft()
            if self._has_untried(cur):
                return True
            for key, edge in self.memory.get(cur, {}).get("edges", {}).items():
                nxt = edge.get("result")
                if not nxt or nxt in seen or key.startswith("RESET"):
                    continue
                if edge.get("state") != GameState.NOT_FINISHED.name:
                    continue
                if edge.get("full_reset") or edge.get("dlevel", 0) < 0:
                    continue
                seen.add(nxt)
                parent[nxt] = cur
                queue.append(nxt)
        return False

    def _is_wedged(self, sig: str) -> bool:
        """True when every known simple action from this state was a no-op.

        Movement games dead-end like this (an avatar that grew too big for the
        corridor); retrying known no-ops only burns the budget.
        """
        node = self.memory.get(sig)
        if not node:
            return False
        simple = [
            edge
            for key, edge in node["edges"].items()
            if not key.startswith("RESET") and "@" not in key
        ]
        return bool(simple) and all(
            not edge.get("changed") and edge.get("dlevel", 0) == 0 for edge in simple
        )

    def _bfs_plan(self, sig: str, remaining: int) -> Optional[list[dict[str, Any]]]:
        """Shortest known path (as action steps) to a state with untried actions."""
        if remaining <= self.RESERVE_ACTIONS + 1:
            return None
        parent: dict[str, tuple[str, str]] = {}
        seen = {sig}
        queue: deque[str] = deque([sig])
        while queue:
            cur = queue.popleft()
            if cur != sig and self._has_untried(cur):
                steps: list[dict[str, Any]] = []
                c = cur
                while c != sig:
                    prev, key = parent[c]
                    steps.append({"from": prev, "key": key, "expect": c})
                    c = prev
                steps.reverse()
                if len(steps) > remaining - self.RESERVE_ACTIONS - 1:
                    return None  # frontier known but too expensive to reach
                return steps
            node = self.memory.get(cur)
            if not node:
                continue
            for key, edge in node["edges"].items():
                nxt = edge.get("result")
                if not nxt or nxt in seen or key.startswith("RESET"):
                    continue
                if edge.get("state") != GameState.NOT_FINISHED.name:
                    continue
                if edge.get("full_reset") or edge.get("dlevel", 0) < 0:
                    continue
                seen.add(nxt)
                parent[nxt] = (cur, key)
                queue.append(nxt)
        return None

    # -------------------------------------------------------------- bookkeeping

    def _record_result(
        self, pending: dict[str, Any], latest: FrameData, sig: Optional[str]
    ) -> None:
        frm = pending["sig"]
        key = pending["key"]
        node = self.memory.setdefault(
            frm, {"avail": [], "edges": {}, "visits": 0, "cands": [], "stuck": 0}
        )
        edge = node["edges"].setdefault(key, {"tries": 0})
        empty = latest.is_empty()
        changed = sig is not None and sig != frm
        edge["tries"] = int(edge.get("tries", 0)) + 1
        edge["dlevel"] = int(latest.levels_completed) - int(pending["levels"])
        edge["state"] = latest.state.name
        edge["full_reset"] = bool(latest.full_reset)
        edge["changed"] = changed
        edge["result"] = None if empty else sig
        if (
            key.startswith("ACTION7")
            and changed
            and not empty
            and sig in pending.get("seen", set())
        ):
            edge["undo"] = True  # ACTION7 restored a previously seen state
        if empty:
            return
        if changed or edge["dlevel"] != 0:
            self.no_change_streak = 0
            aid = self._key_action_id(key)
            if aid is not None:
                self.action_nop_streak[aid] = 0
        else:
            self.no_change_streak += 1
            aid = self._key_action_id(key)
            if aid is not None:
                self.action_nop_streak[aid] = self.action_nop_streak.get(aid, 0) + 1

    def _issue(
        self,
        sig: str,
        latest: FrameData,
        aid: int,
        x: Optional[int],
        y: Optional[int],
        why: str,
    ) -> GameAction:
        self._reset_pending = False
        action = GameAction.from_id(int(aid))
        if x is not None:
            action.set_data({"x": int(x), "y": int(y)})
        action.reasoning = {"agent": "explorer", "why": why}
        self.pending = {
            "sig": sig,
            "key": self._key(aid, x, y),
            "levels": int(latest.levels_completed),
            "seen": set(self._seen),
        }
        return action

    def _give_up_now(self, why: str) -> None:
        if not self._give_up:
            logger.info(f"{self.game_id}: explorer giving up: {why}")
        self._give_up = True

    def _reset_or_giveup(
        self,
        latest: FrameData,
        why: str,
        count_attempt: bool = True,
        sig: Optional[str] = None,
    ) -> GameAction:
        """RESET sparingly; give up when a level is provably not worth more tries."""
        levels = int(latest.levels_completed) if latest is not None else 0
        attempts = self.level_attempts.get(levels, 0)
        if attempts >= self.MAX_LEVEL_ATTEMPTS and not self._level_has_frontier(
            sig, latest
        ):
            # retried often enough and the memory graph offers no unexplored
            # (state, action) pair any more: stop honestly
            self._give_up_now(
                f"level {levels}: attempts exhausted, no frontier left ({why})"
            )
            return GameAction.RESET
        if count_attempt and self.total_levels:
            # Budget pacing: repeated failed attempts on a level that already
            # consumed many times its share of the global budget are not worth
            # another try (levels cannot be skipped, so this only stops futile
            # grinding, never blocks a reachable level).
            share = max(8, self.MAX_ACTIONS // max(1, self.total_levels))
            spent = self.action_counter - self.level_start_action
            if attempts >= 2 and spent > 8 * share:
                self._give_up_now(
                    f"level {levels}: spent {spent} actions (> 5x share {share}) ({why})"
                )
                return GameAction.RESET
        if count_attempt:
            self.level_attempts[levels] = attempts + 1
        self.plan = []
        self.no_change_streak = 0
        self.level_start_action = self.action_counter
        self._reset_pending = True
        logger.info(
            f"{self.game_id}: RESET ({why}); level {levels} attempt "
            f"{self.level_attempts.get(levels, attempts + 1)}"
        )
        return GameAction.RESET

    # -------------------------------------------------------------- agent API

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Done on WIN, exhausted action budget, wall-clock timeout, or when wedged."""
        timer = getattr(self, "timer", None)
        if timer is not None and timer > 0:
            if (time.time() - timer) >= getattr(self, "TIME_LIMIT_S", self.DEFAULT_TIME_LIMIT_S):
                logger.warning(
                    f"{self.game_id}: reached wall-clock limit of {self.TIME_LIMIT_S}s, stopping."
                )
                return True
        return bool(
            latest_frame.state is GameState.WIN
            or self.action_counter >= self.MAX_ACTIONS
            or self._give_up
        )

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        self._update_total_levels(latest_frame)
        empty = latest_frame.is_empty()
        sig: Optional[str] = None
        if not empty:
            self._observe_transition(
                self._grid(latest_frame), int(latest_frame.levels_completed)
            )
            sig = self._signature(latest_frame)

        # Attribute the previous action's observed effect to memory.
        if self.pending is not None:
            self._record_result(self.pending, latest_frame, sig)
            self.pending = None

        # 1) Terminal states: never reset a won game.
        if latest_frame.state is GameState.WIN:
            return GameAction.RESET
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER) or empty:
            first_reset = (
                latest_frame.state is GameState.NOT_PLAYED and self.action_counter == 0
            )
            return self._reset_or_giveup(
                latest_frame,
                f"state {latest_frame.state.name}",
                count_attempt=not first_reset,
                sig=sig,
            )

        levels = int(latest_frame.levels_completed)
        if levels != self.last_levels:
            if self.last_levels >= 0:
                logger.info(
                    f"{self.game_id}: level {self.last_levels} -> {levels} "
                    f"at action {self.action_counter}"
                )
                self.plan = []
                self.no_change_streak = 0
                self.level_start_action = self.action_counter
            self.last_levels = levels

        assert sig is not None
        self._seen.add(sig)
        self.recent.append(sig)
        self.last_sig = sig
        node = self._node(sig, latest_frame)
        node["visits"] += 1

        remaining = self.MAX_ACTIONS - self.action_counter
        if remaining <= self.RESERVE_ACTIONS:
            # With the budget nearly gone, only a known level-completing move
            # is worth an action; plans and probes are budget-checked anyway.
            prog = self._progress_action(sig)
            if prog is not None:
                return self._issue(sig, latest_frame, *prog, why="last-chance progress")

        # 2) Known progress: an action from here that completed a level before.
        prog = self._progress_action(sig)
        if prog is not None:
            return self._issue(sig, latest_frame, *prog, why="known progress action")

        # 3) No-effect churn (possible hidden step budget): reset early rather
        #    than burn the budget, but only when local probing is exhausted.
        if (
            not self.plan
            and self.no_change_streak >= self.NO_CHANGE_LIMIT
            and not self._untried_at(sig)
        ):
            self.no_change_streak = 0
            return self._reset_or_giveup(latest_frame, "no-effect streak", sig=sig)

        # 4) Frontier probing of untried actions from the current state.
        untried = self._untried_at(sig)
        if untried:
            aid, x, y = untried[0]
            return self._issue(sig, latest_frame, aid, x, y, why="frontier probe")

        # 5) Replay a known path to the nearest state with untried actions.
        if self.plan:
            step = self.plan[0]
            if step["from"] == sig:
                self.plan.pop(0)
                aid, x, y = self._parse_key(step["key"])
                return self._issue(
                    sig, latest_frame, aid, x, y, why="replay to frontier"
                )
            self.plan = []  # replay drifted off the recorded path; re-plan

        plan = self._bfs_plan(sig, remaining)
        if plan:
            self.plan = plan
            step = plan.pop(0)
            aid, x, y = self._parse_key(step["key"])
            return self._issue(sig, latest_frame, aid, x, y, why="replay to frontier")

        # 6) Stuck / cycling: least-recently-tried action, then a sparing
        #    level RESET, then stop honestly.
        if list(self.recent).count(sig) >= 3:
            logger.debug(f"{self.game_id}: cycle detected at {sig}")
        wedged = self._is_wedged(sig)
        if node.get("stuck", 0) < self.MAX_STUCK_LRU and not wedged:
            node["stuck"] = int(node.get("stuck", 0)) + 1
            lru = self._lru_action(sig)
            if lru is not None:
                return self._issue(
                    sig, latest_frame, *lru, why="cycle break (least-recently-tried)"
                )
        return self._reset_or_giveup(
            latest_frame, "no frontier reachable", sig=sig
        )


# Canonical alias for Kaggle submission pipeline compatibility
MyAgent = Explorer

__all__ = ["Explorer", "MyAgent"]
