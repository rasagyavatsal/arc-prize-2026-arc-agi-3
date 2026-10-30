"""Build our submission notebook from Daniel Franzen's public Milestone 2 solution.

https://www.kaggle.com/code/dfranzen/arc-agi-3-milestone-2-solution (Apache-2.0,
https://github.com/da-fr/arc-agi-3-solution): Tufa Labs' Duck harness, where a
local Qwen3.8-Flash-Next served by SGLang on the RTX Pro 6000 plays every game.
Third-party code is not copied into this repo: the notebook is downloaded at
build time and patched here.

The competition rerun (`KAGGLE_IS_COMPETITION_RERUN`) is left exactly as
Franzen set it.  Only the interactive "Save & Run" changes: his plays 4 passes
of the 25 demo games for ~9 GPU hours just to emit the placeholder
submission.parquet a version needs before it can be submitted.  Ours plays
QUICK_GAMES games for QUICK_MINUTES as an end-to-end check (server up, games
played, file written).

Usage (from ARC-AGI-3-Kaggle-Starter/):
  kaggle kernels pull dfranzen/arc-agi-3-milestone-2-solution -p notebooks/franzen-src -m
  python scripts/build_franzen_notebook.py            # writes notebooks/franzen/
  kaggle kernels push -p notebooks/franzen --accelerator NvidiaRtxPro6000
  kaggle competitions submit arc-prize-2026-arc-agi-3 -k rasagyavatsal/arc-agi-3-franzen \\
      -v <version> -f submission.parquet -m "..."
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "notebooks" / "franzen-src"
OUT_DIR = ROOT / "notebooks" / "franzen"

KERNEL_ID = "rasagyavatsal/arc-agi-3-franzen"
TITLE = "ARC-AGI-3 Franzen"
ACCELERATOR = "NvidiaRtxPro6000"  # Kaggle silently falls back to a T4 on an unknown name
QUICK_GAMES = 3
QUICK_MINUTES = 25

QUICK_CONFIG = f"""\
# Interactive run only: a short end-to-end check instead of ~9 GPU hours.
# The competition rerun ignores these.
QUICK_GAMES = {QUICK_GAMES}
QUICK_MINUTES = {QUICK_MINUTES}
"""

# (cell marker, old text, new text): each old text must occur exactly once
PATCHES = [
    ("bm.solver.max_runtime_s_per_game = 48*60",
     """    if USE_PRIORITY_SCHEDULING:
        bm.solver.concurrency = (25 - len(demo_excluded_games)) * bm.n_passes
        bm.solver.max_runtime_s_per_game = 532*60 * bm.solver.concurrency // 110
""",
     """    bm.n_passes = 1
    bm.solver.concurrency = QUICK_GAMES
    bm.solver.max_runtime_s_per_game = QUICK_MINUTES * 60
"""),
    ("print('Starting benchmark...')",
     """    bm.games = _offline_games(competition_env_files)
""",
     """    bm.games = _offline_games(competition_env_files)[:QUICK_GAMES]
"""),
    ("print('Starting benchmark...')",
     """        soft_end = datetime.fromtimestamp(NOTEBOOK_START_EPOCH) + timedelta(seconds=budget - min(600.0, budget / 2))
""",
     """        soft_end = datetime.fromtimestamp(NOTEBOOK_START_EPOCH) + timedelta(seconds=budget - min(600.0, budget / 2))
    soft_end = min(soft_end or datetime.max, datetime.now() + timedelta(minutes=QUICK_MINUTES + 5))
"""),
]


def find_cell(cells: list, marker: str) -> dict:
    hits = [c for c in cells if c["cell_type"] == "code" and marker in "".join(c["source"])]
    if len(hits) != 1:
        raise SystemExit(f"expected one cell containing {marker!r}, found {len(hits)}: "
                         "the source notebook changed, re-check the patches")
    return hits[0]


def build(src_dir: Path) -> tuple[dict, dict]:
    nb = json.loads(next(src_dir.glob("*.ipynb")).read_text())
    meta = json.loads((src_dir / "kernel-metadata.json").read_text())
    cells = nb["cells"]
    for marker, old, new in PATCHES:
        cell = find_cell(cells, marker)
        src = "".join(cell["source"])
        if src.count(old) != 1:
            raise SystemExit(f"patch text not found once in the {marker!r} cell:\n{old}")
        cell["source"] = src.replace(old, new)
    for cell in cells:
        if cell["cell_type"] == "code":
            cell["outputs"], cell["execution_count"] = [], None
        else:  # nbformat rejects outputs on markdown cells
            cell.pop("outputs", None)
            cell.pop("execution_count", None)
    cells.insert(0, {"cell_type": "code", "metadata": {"trusted": True}, "outputs": [],
                     "execution_count": None, "source": QUICK_CONFIG})
    cells.insert(0, {"cell_type": "markdown", "metadata": {}, "source":
                     "Daniel Franzen's [ARC-AGI-3 Milestone 2 Solution]"
                     "(https://www.kaggle.com/code/dfranzen/arc-agi-3-milestone-2-solution) "
                     "(Apache-2.0), built on Tufa Labs' Duck harness. Unchanged for the "
                     "competition rerun; the interactive run is shortened to a quick check."})
    out_meta = {
        "id": KERNEL_ID,
        "title": TITLE,
        "code_file": "franzen.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": meta["dataset_sources"],
        "kernel_sources": meta.get("kernel_sources", []),
        "competition_sources": meta["competition_sources"],
        "model_sources": meta["model_sources"],
        "docker_image": meta["docker_image"],
        "machine_shape": ACCELERATOR,
    }
    return nb, out_meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=SRC_DIR, help="pulled Franzen notebook dir")
    args = ap.parse_args()
    nb, meta = build(args.src)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / meta["code_file"]).write_text(json.dumps(nb, indent=1))
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {OUT_DIR} ({KERNEL_ID}); push with --accelerator {ACCELERATOR}")


if __name__ == "__main__":
    main()
