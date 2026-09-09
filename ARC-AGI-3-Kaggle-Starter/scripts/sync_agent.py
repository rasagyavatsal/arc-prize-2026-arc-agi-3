"""Synchronize and verify single source of truth between Explorer and MyAgent.

Ensures ARC-AGI-3-Kaggle-Starter/agent/my_agent.py and
ARC-AGI-3-Agents/agents/templates/explorer.py are identical and unified.
"""
from __future__ import annotations

import filecmp
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # ARC-AGI-3-Kaggle-Starter
WORKSPACE = ROOT.parent
MY_AGENT = ROOT / "agent" / "my_agent.py"
EXPLORER = WORKSPACE / "ARC-AGI-3-Agents" / "agents" / "templates" / "explorer.py"


def sync(check_only: bool = False) -> bool:
    global EXPLORER
    if not EXPLORER.exists():
        vendor_explorer = ROOT / "vendor" / "ARC-AGI-3-Agents" / "agents" / "templates" / "explorer.py"
        if vendor_explorer.exists():
            EXPLORER = vendor_explorer
        else:
            print(f"[sync_agent] ERROR: Canonical agent not found at {EXPLORER}")
            return False

    MY_AGENT.parent.mkdir(parents=True, exist_ok=True)

    if MY_AGENT.is_symlink():
        target = os.readlink(MY_AGENT)
        resolved = (MY_AGENT.parent / target).resolve()
        if resolved == EXPLORER.resolve():
            print(f"[sync_agent] OK: my_agent.py is symlinked to canonical explorer.py ({target})")
            return True
        print(f"[sync_agent] WARNING: my_agent.py symlink points to {target} instead of {EXPLORER}")

    # Check content equality
    if MY_AGENT.exists():
        if filecmp.cmp(MY_AGENT, EXPLORER, shallow=False):
            print("[sync_agent] OK: my_agent.py content perfectly matches explorer.py")
            return True
        if check_only:
            print("[sync_agent] FAIL: my_agent.py differs from explorer.py")
            return False

    if check_only:
        print("[sync_agent] FAIL: my_agent.py is missing or not unified")
        return False

    # Establish symlink with copy fallback (e.g. Windows without dev privileges)
    if MY_AGENT.exists() or MY_AGENT.is_symlink():
        MY_AGENT.unlink()

    rel_target = os.path.relpath(EXPLORER, MY_AGENT.parent)
    try:
        os.symlink(rel_target, MY_AGENT)
        print(f"[sync_agent] Successfully unified: created symlink {MY_AGENT} -> {rel_target}")
    except OSError as e:
        import shutil
        shutil.copyfile(EXPLORER, MY_AGENT)
        print(f"[sync_agent] Symlink not permitted ({e}), unified via copy: {MY_AGENT} <= {EXPLORER}")
    return True


if __name__ == "__main__":
    check_mode = "--check" in sys.argv
    success = sync(check_only=check_mode)
    sys.exit(0 if success else 1)
