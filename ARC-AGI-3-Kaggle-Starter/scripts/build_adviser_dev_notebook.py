"""Build the adviser dev-run notebook: Explorer with vs without GPT-OSS-120B.

Not a submission.  The notebook plays the dev split (`sweeps/heldout_eval.py
--split dev`) on Kaggle's RTX Pro 6000, offline, twice:

  1. without the adviser, while vLLM loads in the background;
  2. with the adviser (`ADVISER_URL` -> the local vLLM server),

and in between measures the server: startup time, then latency and
throughput at several concurrency levels on real object-list prompts.
Held-out games are never played here (tune on dev only).

Usage (from ARC-AGI-3-Kaggle-Starter/):
  python scripts/build_adviser_dev_notebook.py      # writes notebooks/adviser-dev/
  kaggle kernels push -p notebooks/adviser-dev --accelerator NvidiaRtxPro6000
"""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
EXPLORER = WORKSPACE / "ARC-AGI-3-Agents" / "agents" / "templates" / "explorer.py"
HELDOUT_EVAL = WORKSPACE / "sweeps" / "heldout_eval.py"
OUT_DIR = ROOT / "notebooks" / "adviser-dev"

KERNEL_ID = "rasagyavatsal/arc-agi-3-adviser-dev"
MODEL_SOURCE = "danielhanchen/gpt-oss-120b/Transformers/default/1"
ACCELERATOR = "NvidiaRtxPro6000"  # Kaggle silently falls back to a T4 on an unknown name
VLLM_DEPS = "philipvonderlind/vllm-deps"  # vLLM wheels + tiktoken files (official template)
# the vllm-deps wheels are cp312: pin the official template's Python 3.12 image
# (the current default image is 3.13, where the vLLM install fails)
DOCKER_IMAGE = "gcr.io/kaggle-images/python@sha256:e5452ce6268c2e8345cfe5141f31ca7ff47032aca46a7ea532bbb87481281d0c"


def code_cell(source: str) -> dict:
    return {"cell_type": "code", "metadata": {"trusted": True}, "outputs": [],
            "execution_count": None, "source": source}


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


CONFIG = dedent(
    """\
    # Run configuration (edit here, not in the cells below)
    MODEL_PATH = "/kaggle/input/models/danielhanchen/gpt-oss-120b/transformers/default/1"
    SERVED_NAME = "gpt-oss-120b"
    PORT = 8000
    MAX_NUM_SEQS = 64          # concurrent sequences: every game may ask at once
    MAX_MODEL_LEN = 8192       # prompts are ~1k tokens, answers <= 1k
    ENFORCE_EAGER = True       # template default; False = CUDA graphs, faster but slower start
    STARTUP_TIMEOUT_S = 2400
    BENCH_CONCURRENCY = [1, 8, 32, 64]
    ADVISER_TIMEOUT_S = 30     # Explorer defaults; raise to see what slower calls buy
    ADVISER_BUDGET_S = 240

    import os, subprocess, sys, time, json
    from pathlib import Path
    T0 = time.time()
    W = Path("/kaggle/working/w")
    COMP = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3")
    # The pinned (older) image may not have the host driver's NVML on its
    # library path; nvidia-smi and vLLM both need it.
    nvml = sorted({str(p.parent) for root in ("/usr/local/nvidia", "/usr/lib", "/usr/lib64",
                                                "/usr/local/cuda", "/opt", "/usr/local/lib")
                   if Path(root).exists() for p in Path(root).rglob("libnvidia-ml.so*")})
    print("libnvidia-ml dirs:", nvml)
    if nvml:
        os.environ["LD_LIBRARY_PATH"] = ":".join(nvml + [os.environ.get("LD_LIBRARY_PATH", "")]).rstrip(":")
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True)
    print(smi.stdout, smi.stderr[-500:])
    # this image's own torch is CPU-only (vLLM installs its CUDA build), so
    # the GPU check reads nvidia-smi
    gpu_mb = max([int(f) for line in smi.stdout.splitlines()
                  for f in line.split(",")[-1:] if f.strip().isdigit()] or [0])
    if gpu_mb < 80000:
        # RuntimeError, not SystemExit: a notebook run carries on after SystemExit
        raise RuntimeError(f"need the ~96 GB RTX Pro 6000, nvidia-smi shows {gpu_mb} MiB: stopping")
    print("cpus", os.cpu_count(), "| python", sys.version.split()[0])
    """
)

INSTALL = dedent(
    """\
    wheels = list(Path("/kaggle/input").rglob("vllm*.whl"))
    if not wheels:
        raise FileNotFoundError("no vllm wheel under /kaggle/input (attach the vllm-deps notebook)")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-index",
                           "--find-links", str(wheels[0].parent),
                           # the image's torch 2.10.0+cpu satisfies vllm's torch pin, so
                           # the CUDA builds must be named or pip keeps the CPU one
                           "torch==2.10.0+cu128", "torchvision==0.25.0+cu128",
                           "torchaudio==2.10.0+cu128",
                           "vllm==0.19.1", "openai==2.24.0", "openai-harmony==0.0.8"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-index",
                           "--find-links", str(COMP / "arc_agi_3_wheels"), "arc-agi", "python-dotenv"])
    TIKTOKEN = next(d for d in Path("/kaggle/input").rglob("*") if d.is_dir()
                    and (d / "cl100k_base.tiktoken").exists() and (d / "o200k_base.tiktoken").exists())
    print("tiktoken", TIKTOKEN, "| install done at", round(time.time() - T0), "s")
    """
)

WORKSPACE_SETUP = dedent(
    """\
    import shutil, textwrap
    shutil.rmtree(W, ignore_errors=True)
    (W / "sweeps").mkdir(parents=True)
    (W / "notes").mkdir()
    shutil.copytree(COMP / "ARC-AGI-3-Agents", W / "ARC-AGI-3-Agents")
    shutil.copy("/tmp/explorer.py", W / "ARC-AGI-3-Agents/agents/templates/explorer.py")
    shutil.copy("/tmp/heldout_eval.py", W / "sweeps/heldout_eval.py")
    (W / "environment_files").symlink_to(COMP / "environment_files")
    # slim registry: upstream __init__ imports templates whose deps are not installed
    (W / "ARC-AGI-3-Agents/agents/__init__.py").write_text(textwrap.dedent('''\\
        from dotenv import load_dotenv
        from .agent import Agent, Playback
        from .swarm import Swarm
        from .templates.explorer import Explorer, MyAgent
        load_dotenv()
        AVAILABLE_AGENTS = {"explorer": Explorer, "myagent": MyAgent}
        '''))
    print(sorted(p.name for p in (COMP / "environment_files").iterdir()))
    """
)

START_VLLM = dedent(
    """\
    import urllib.request
    env = dict(os.environ, TIKTOKEN_ENCODINGS_BASE=str(TIKTOKEN), VLLM_WORKER_MULTIPROC_METHOD="spawn")
    cmd = ["vllm", "serve", MODEL_PATH, "--served-model-name", SERVED_NAME,
           "--host", "127.0.0.1", "--port", str(PORT), "--max-num-seqs", str(MAX_NUM_SEQS),
           "--max-model-len", str(MAX_MODEL_LEN), "--kv-cache-dtype", "fp8"]
    if ENFORCE_EAGER:
        cmd.append("--enforce-eager")
    VLLM_LOG = open("/kaggle/working/vllm.log", "w")
    vllm_proc = subprocess.Popen(cmd, stdout=VLLM_LOG, stderr=subprocess.STDOUT, env=env)
    VLLM_STARTED = time.time()
    print("vLLM starting in the background:", " ".join(cmd))

    def healthy():
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                return r.status == 200
        except Exception:
            return False
    """
)

BASELINE = dedent(
    """\
    # Plain Explorer on dev while the model loads (CPU only)
    run_env = {k: v for k, v in os.environ.items() if not k.startswith("ADVISER_")}
    r = subprocess.run([sys.executable, "sweeps/heldout_eval.py", "--split", "dev",
                        "--out", "/kaggle/working/dev_without_adviser.json"],
                       cwd=W, env=run_env, capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-3000:])
    """
)

WAIT_VLLM = dedent(
    """\
    while not healthy():
        if vllm_proc.poll() is not None:
            VLLM_LOG.flush()
            print(open("/kaggle/working/vllm.log").read()[-8000:])
            raise RuntimeError(f"vLLM exited with {vllm_proc.returncode}")
        if time.time() - VLLM_STARTED > STARTUP_TIMEOUT_S:
            raise TimeoutError("vLLM did not come up")
        time.sleep(5)
    STARTUP_S = round(time.time() - VLLM_STARTED)
    print(f"vLLM healthy after {STARTUP_S}s")
    """
)

BENCH = dedent(
    """\
    # Latency / throughput on real prompts: each dev game's level-0 start frame
    import statistics, urllib.request
    from concurrent.futures import ThreadPoolExecutor
    sys.path.insert(0, str(W / "ARC-AGI-3-Agents"))
    os.environ.update(OPERATION_MODE="offline", ENVIRONMENTS_DIR=str(W / "environment_files"))
    from arc_agi import Arcade
    from agents.templates import explorer as ex

    prompts = []
    for game in ["cd82", "g50t", "lp85", "ls20", "m0r0", "r11l", "s5i5", "sp80", "tu93", "vc33"]:
        env = Arcade().make(game)
        agent = ex.Explorer(card_id="bench", game_id=game, agent_name="bench", ROOT_URL="",
                            record=False, arc_env=env, adviser=ex.Adviser())
        frame = agent._convert_raw_frame_data(env.reset())
        prompts.append(agent._advice_prompt(frame, agent._prompt_objects(agent._grid(frame))))

    def call(prompt):
        body = {"model": SERVED_NAME, "max_tokens": 1024, "temperature": 0, "reasoning_effort": "low",
                "messages": [{"role": "system", "content": ex.ADVISER_SYSTEM},
                             {"role": "user", "content": json.dumps(prompt, separators=(",", ":"))}]}
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t = time.time()
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read())
            text = data["choices"][0]["message"].get("content")
            usage = data.get("usage", {})
        except Exception as e:
            return {"s": time.time() - t, "ok": False, "error": repr(e)}
        return {"s": time.time() - t, "ok": ex.parse_advice(text) is not None, "text": text,
                "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens")}

    call(prompts[0])  # warm-up
    BENCH = {"startup_s": STARTUP_S}
    for n in BENCH_CONCURRENCY:
        batch = [prompts[i % len(prompts)] for i in range(n)]
        t = time.time()
        with ThreadPoolExecutor(n) as pool:
            res = list(pool.map(call, batch))
        wall = time.time() - t
        lat = sorted(r["s"] for r in res)
        BENCH[n] = {"calls_per_min": round(60 * n / wall, 1), "p50_s": round(statistics.median(lat), 1),
                    "max_s": round(lat[-1], 1), "parsed": sum(r["ok"] for r in res),
                    "completion_tokens_mean": round(statistics.mean(r.get("completion_tokens") or 0 for r in res)),
                    "prompt_tokens_mean": round(statistics.mean(r.get("prompt_tokens") or 0 for r in res))}
        print(n, BENCH[n])
    for r in res[:3]:
        print(repr(r.get("text") or r.get("error"))[:600])
    json.dump(BENCH, open("/kaggle/working/bench.json", "w"), indent=1)
    """
)

ADVISED = dedent(
    """\
    run_env = dict(os.environ, ADVISER_URL=f"http://127.0.0.1:{PORT}/v1", ADVISER_MODEL=SERVED_NAME,
                   ADVISER_TIMEOUT_S=str(ADVISER_TIMEOUT_S), ADVISER_BUDGET_S=str(ADVISER_BUDGET_S))
    r = subprocess.run([sys.executable, "sweeps/heldout_eval.py", "--split", "dev",
                        "--out", "/kaggle/working/dev_with_adviser.json"],
                       cwd=W, env=run_env, capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-3000:])
    """
)

SUMMARY = dedent(
    """\
    out = {"bench": BENCH}
    for name in ("without", "with"):
        path = Path(f"/kaggle/working/dev_{name}_adviser.json")
        out[name] = json.load(open(path))["summary"] if path.exists() else None
    out["total_s"] = round(time.time() - T0)
    json.dump(out, open("/kaggle/working/summary.json", "w"), indent=1)
    print(json.dumps(out, indent=1))
    vllm_proc.terminate()
    """
)


def build() -> dict:
    return {
        "metadata": {
            "kernelspec": {"language": "python", "display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {"accelerator": "nvidiaRtx6000", "isInternetEnabled": False,
                       "isGpuEnabled": True, "language": "python", "sourceType": "notebook"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
        "cells": [
            markdown_cell(
                "# ARC-AGI-3 adviser dev run\n\nExplorer with vs without a GPT-OSS-120B adviser on "
                "the dev split. Built by `scripts/build_adviser_dev_notebook.py`; edit that, not "
                "the cells."
            ),
            code_cell(CONFIG),
            code_cell("%%writefile /tmp/explorer.py\n" + EXPLORER.read_text()),
            code_cell("%%writefile /tmp/heldout_eval.py\n" + HELDOUT_EVAL.read_text()),
            code_cell(INSTALL),
            code_cell(WORKSPACE_SETUP),
            code_cell(START_VLLM),
            code_cell(BASELINE),
            code_cell(WAIT_VLLM),
            code_cell(BENCH),
            code_cell(ADVISED),
            code_cell(SUMMARY),
        ],
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "adviser-dev.ipynb").write_text(json.dumps(build(), indent=1))
    meta = {
        "id": KERNEL_ID,
        "title": "ARC-AGI-3 Adviser Dev",
        "code_file": "adviser-dev.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [],
        "kernel_sources": [VLLM_DEPS],
        "competition_sources": ["arc-prize-2026-arc-agi-3"],
        "model_sources": [MODEL_SOURCE],
        "machine_shape": ACCELERATOR,
        "docker_image": DOCKER_IMAGE,
    }
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[build_adviser_dev_notebook] wrote {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
