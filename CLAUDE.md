# CLAUDE.md

Dataset production for [transformer-oc](https://github.com/SegmentLinking/transformer-oc) (ML track reconstruction for CMS Phase-2, PU200). The pipeline is step2 RAW → tracking ntuple → LSTNtuple → one PyTorch Geometric graph of LST T3 objects per event. See `README.md` for the stages, usage and graph format.

## Layout

- `cmssw/`: bash scripts for the CMSSW stages. No pip dependencies: each script sets up CMSSW from `/cvmfs` and the release area in `CMSSW_SRC` if its tool (`cmsRun`, `lst_cpu`) isn't on the `PATH`. All are safe to rerun: existing outputs are skipped.
- `samples/<sample>/`: file lists (`relval.txt` for step2 inputs, `trackingntuple.txt` for stage 1 outputs). Keep them as absolute paths, one per line; they record which files went into each dataset.
- `src/transformer_datasets/`: the installable Python package. `t3_processing.py` is the graph builder, `constants.py` holds ntuple column names and cuts. `md_processing.py` is an experimental MD-level alternative: plain dicts of tensors, no PyG, with its own column lists.
- `tests/`: pytest tests (plain asserts; the files also run as scripts without pytest).

## Environment

- CMSSW stages: release area `/ceph/users/atuna/CMSSW_20_1_0_pre2/src`, with LST standalone built under `RecoTracker/LSTCore/standalone`.
- Python stage: a normal installable package, independent of any container. Install it editable into a venv in the repo root and run it as a module:

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -e .
python -m transformer_datasets.t3_processing --help
pytest
```

Dependencies are declared in `pyproject.toml`; add new ones there rather than assuming they're installed. Graph building runs on CPU.

## Conventions

- Use package imports (`from transformer_datasets.constants import ...`), not flat imports or `sys.path` hacks.
- Format Python with black. One argument per line with a trailing comma when a call or signature doesn't fit on one line, keyword arguments where possible, and type hints on all signatures.
- Scripts read top-down: `main` comes first, as a short outline (about 10–20 lines), followed by the functions it calls in roughly call order. Argument parsing lives in `parse_args()`, and the `args` namespace is unpacked in `main` and never passed to other functions.
- Prefer explicit named sets in constants (e.g. `BARREL_LAYERS = range(1, 7)`) over thresholds like `layer <= 6`.
- Bash scripts: `set -euo pipefail`, configuration through environment variables with defaults, `[info]`/`[done]`-style log lines, and write to a temporary name and rename on success so interrupted jobs don't leave outputs that look finished.
- Focus on T3 graphs. The LS-graph variant from transformer-oc was intentionally not carried over.

## Gotchas

- **The graph format is a contract with transformer-oc.** Its `src/constants.py` and `src/dataset.py` hardcode the 49-column layout of `x` (normalization ranges, η/φ in columns 1 and 2). Changing features here needs a matching change there.
- `t3_processing.py` checks its recomputed `t3_pMatched` against the ntuple and raises on mismatch; the exception is caught per event and printed, so a failing event is skipped rather than stopping the job.
- `md_anchor_*` is not always the inner hit of an MD. `md_processing.py` orders hits by rT in the barrel and |z| in the endcap.
- `trackingntuple_maker.sh`'s `PILEUP_INPUT` must match the pileup the step2 files were mixed with.

## Workflow

- Develop on a branch and PR into `SegmentLinking/transformer-datasets` `main`. Don't commit or push without being asked.
- Don't commit data: `.root`, `.pt`, logs or generated CMSSW configs (`step3_*.py`).
- Run the tests and show the exact command used.
