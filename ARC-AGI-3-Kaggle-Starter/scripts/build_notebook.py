"""Splice the current `agent/my_agent.py` into `notebooks/submission.ipynb`.

The notebook follows the exact pattern used by Kaggle's official sample
("ARC3 Sample Submission - Stochastic Goose"):

  Cell 1: install the `arc-agi` wheel from the offline competition dataset.
  Cell 2: write `my_agent.py` to /kaggle/working/ — its body is THIS file.
  Cell 3: if running inside the Kaggle competition rerun, wait for the
          gateway sidecar, copy the framework into /kaggle/working/, register
          MyAgent, and run `python main.py --agent myagent`.
  Cell 4: otherwise (during commit / save-and-run-all), write a dummy
          submission.parquet so Kaggle accepts the commit.

You don't normally need to call this directly — `make submit` runs it for you.
"""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

# ─────────────────────────────────────────────────────────────────────────────
# CHANGE THIS ONE LINE TO PICK YOUR KAGGLE ACCELERATOR
# Options:
#   "cpu"      — no GPU. Good for the random starter or any non-ML agent.
#   "t4"       — Nvidia T4 ×2 (default; matches Kaggle's sample submission).
#   "p100"     — Nvidia P100 (single big-memory GPU).
#   "rtx6000"  — Nvidia RTX 6000 (g4-standard-48). ARC-AGI-3 exclusive,
#                burns GPU quota faster — use only when you're confident.
# ─────────────────────────────────────────────────────────────────────────────
ACCELERATOR = "cpu"   # Explorer is deterministic CPU-only; flip to t4/rtx6000 for model-based agents

# LLM adviser (issue #7): serve GPT-OSS-120B with vLLM next to the Explorer and
# point it there via ADVISER_URL. Needs ACCELERATOR = "rtx6000". The server
# loads in the background during the rerun; games play plain Explorer until
# it is healthy, so the first-action deadline is never at risk.
ADVISER = False
ADVISER_MODEL_SOURCE = "danielhanchen/gpt-oss-120b/Transformers/default/1"
ADVISER_KERNEL_SOURCE = "philipvonderlind/vllm-deps"  # vLLM wheels + tiktoken files

# Internal mapping; don't edit unless Kaggle adds new options.
_ACCELERATORS = {
    "cpu":     {"name": "none",            "gpu": False},
    "t4":      {"name": "nvidiaTeslaT4",   "gpu": True},
    "p100":    {"name": "nvidiaTeslaP100", "gpu": True},
    "rtx6000": {"name": "nvidiaRtx6000",   "gpu": True},
}
# machine_shape names for kernel-metadata.json; Kaggle silently runs a T4 on
# an unknown name
_SHAPES = {"cpu": "None", "t4": "NvidiaTeslaT4", "p100": "NvidiaTeslaP100",
           "rtx6000": "NvidiaRtxPro6000"}

ROOT = Path(__file__).resolve().parents[1]
AGENT_SRC = ROOT / "agent" / "my_agent.py"
NOTEBOOK_PATH = ROOT / "notebooks" / "submission.ipynb"
METADATA_PATH = ROOT / "notebooks" / "kernel-metadata.json"


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {"trusted": True},
        "outputs": [],
        "execution_count": None,
        "source": source,
    }


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def build() -> dict:
    if not AGENT_SRC.exists():
        raise SystemExit(f"Could not find {AGENT_SRC}")
    agent_body = AGENT_SRC.read_text()

    install_cell = code_cell(
        "!pip install --no-index --find-links \\\n"
        "    /kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels \\\n"
        "    arc-agi python-dotenv"
    )

    # We write the agent to /tmp/ (not /kaggle/working/) so it does NOT appear
    # as a notebook output. Otherwise the "Submit to Competition" UI would
    # offer it as a candidate submission file alongside submission.parquet,
    # and an unlucky default selection rejects the submission.
    write_agent_cell = code_cell(
        "%%writefile /tmp/my_agent.py\n" + agent_body
    )

    run_cell_source = dedent(
        """\
        import os

        if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
            # Wait for the gateway sidecar to be ready.
            !curl --fail --retry 999 --retry-all-errors --retry-delay 5 \\
                  --retry-max-time 600 http://gateway:8001/api/games

            # Copy the framework into a writable location.
            !cp -r /kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents \\
                   /kaggle/working/ARC-AGI-3-Agents

            # Drop our agent in as a framework template.
            !cp /tmp/my_agent.py \\
                /kaggle/working/ARC-AGI-3-Agents/agents/templates/my_agent.py

            # Register MyAgent in the framework's agent registry. We rewrite
            # __init__.py because the upstream version eagerly imports
            # templates with deps we don't ship (langgraph, smolagents, etc.).
            with open('/kaggle/working/ARC-AGI-3-Agents/agents/__init__.py', 'w') as f:
                f.write(\"\"\"from typing import Type
        from dotenv import load_dotenv
        from .agent import Agent, Playback
        from .swarm import Swarm
        from .templates.random_agent import Random
        from .templates.my_agent import MyAgent

        load_dotenv()

        AVAILABLE_AGENTS: dict[str, Type[Agent]] = {
            'random': Random,
            'myagent': MyAgent,
        }
        \"\"\")

            # Point the framework at the gateway sidecar.
            with open('/kaggle/working/ARC-AGI-3-Agents/.env', 'w') as f:
                f.write(\"\"\"SCHEME=http
        HOST=gateway
        PORT=8001
        ARC_API_KEY=test-key-123
        ARC_BASE_URL=http://gateway:8001/
        OPERATION_MODE=online
        ENVIRONMENTS_DIR=
        RECORDINGS_DIR=/kaggle/working/server_recording
        \"\"\")

            # Run it. The gateway records every action and emits submission.parquet.
            !cd /kaggle/working/ARC-AGI-3-Agents && \\
                MPLBACKEND=agg \\
                python main.py --agent myagent
        """
    )
    run_cell = code_cell(run_cell_source)

    adviser_cell = code_cell(
        dedent(
            """\
            import os
            if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
                # Start vLLM in the background; the Explorer polls /health and
                # uses the adviser only once the model is up.
                import subprocess
                from pathlib import Path
                wheels = list(Path('/kaggle/input').rglob('vllm*.whl'))
                tiktoken = [d for d in Path('/kaggle/input').rglob('*') if d.is_dir()
                            and (d / 'cl100k_base.tiktoken').exists()
                            and (d / 'o200k_base.tiktoken').exists()]
                if wheels and tiktoken:
                    subprocess.run(['pip', 'install', '-q', '--no-index', '--find-links',
                                    str(wheels[0].parent), 'vllm==0.19.1', 'openai==2.24.0',
                                    'openai-harmony==0.0.8'])
                    env = dict(os.environ, TIKTOKEN_ENCODINGS_BASE=str(tiktoken[0]),
                               VLLM_WORKER_MULTIPROC_METHOD='spawn')
                    subprocess.Popen(
                        ['vllm', 'serve',
                         '/kaggle/input/models/danielhanchen/gpt-oss-120b/transformers/default/1',
                         '--served-model-name', 'gpt-oss-120b', '--host', '127.0.0.1',
                         '--port', '8000', '--max-num-seqs', '64', '--max-model-len', '8192',
                         '--kv-cache-dtype', 'fp8', '--enforce-eager'],
                        stdout=open('/tmp/vllm.log', 'w'), stderr=subprocess.STDOUT, env=env)
                    os.environ['ADVISER_URL'] = 'http://127.0.0.1:8000/v1'
                    os.environ['ADVISER_MODEL'] = 'gpt-oss-120b'
                else:
                    print('adviser inputs missing: plain Explorer')
            """
        )
    )

    dummy_submission_cell = code_cell(
        dedent(
            """\
            import os
            if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
                # Save-and-run-all (commit) mode: emit a dummy submission so the
                # commit succeeds. The real submission.parquet is produced by the
                # gateway during competition rerun.
                import pandas as pd
                submission = pd.DataFrame(
                    data=[['1_0', '1', True, 1]],
                    columns=['row_id', 'game_id', 'end_of_game', 'score'])
                submission.to_parquet('/kaggle/working/submission.parquet', index=False)
                submission.head()
            """
        )
    )

    if ACCELERATOR not in _ACCELERATORS:
        raise SystemExit(
            f"Unknown ACCELERATOR={ACCELERATOR!r}. Pick one of: "
            f"{sorted(_ACCELERATORS)}"
        )
    accel = _ACCELERATORS[ACCELERATOR]
    if ADVISER and ACCELERATOR != "rtx6000":
        raise SystemExit("ADVISER needs ACCELERATOR = 'rtx6000' (GPT-OSS-120B needs ~80 GB)")

    notebook = {
        "metadata": {
            "kernelspec": {
                "language": "python",
                "display_name": "Python 3",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "mimetype": "text/x-python",
                "file_extension": ".py",
                "pygments_lexer": "ipython3",
            },
            "kaggle": {
                "accelerator": accel["name"],
                "isInternetEnabled": False,
                "isGpuEnabled": accel["gpu"],
                "language": "python",
                "sourceType": "notebook",
            },
        },
        "nbformat_minor": 4,
        "nbformat": 4,
        "cells": [
            markdown_cell(
                "# ARC Prize 2026 — ARC-AGI-3 Submission\n\n"
                "Built from `agent/my_agent.py` via `scripts/build_notebook.py`. "
                "Do not edit cells directly — edit the source file and re-run "
                "`make submit`."
            ),
            install_cell,
            write_agent_cell,
            *([adviser_cell] if ADVISER else []),
            run_cell,
            dummy_submission_cell,
        ],
    }
    return notebook


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(build(), indent=1))
    print(f"[build_notebook] Wrote {NOTEBOOK_PATH.relative_to(ROOT)}  "
          f"(accelerator: {ACCELERATOR})")

    # Keep notebooks/kernel-metadata.json in sync so the user never has to
    # edit it just to flip CPU ↔ GPU.
    if METADATA_PATH.exists():
        meta = json.loads(METADATA_PATH.read_text())
        wanted = {
            "enable_gpu": _ACCELERATORS[ACCELERATOR]["gpu"],
            "machine_shape": _SHAPES[ACCELERATOR],
            "kernel_sources": [ADVISER_KERNEL_SOURCE] if ADVISER else [],
            "model_sources": [ADVISER_MODEL_SOURCE] if ADVISER else [],
        }
        changed = {k: v for k, v in wanted.items() if meta.get(k) != v}
        if changed:
            meta.update(changed)
            METADATA_PATH.write_text(json.dumps(meta, indent=2) + "\n")
            print(f"[build_notebook] Synced {changed} in "
                  f"{METADATA_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
