# transformer-datasets

Datasets for [transformer-oc](https://github.com/SegmentLinking/transformer-oc): CMS Phase-2 (PU200) simulation turned into one PyTorch Geometric graph per event.

## Pipeline

```
step2 RAW (GEN-SIM-DIGI-RAW)
    │  cmssw/trackingntuple_maker.sh     (CMSSW, cmsRun)
    ▼
trackingntuple_<name>.root
    │  cmssw/lstntuple_maker.sh          (CMSSW, LST standalone)
    ▼
LSTNtuple_<name>.root
    │  transformer_datasets.t3_processing (Python)
    ▼
{train,val}/graph_<N>.pt
```

The first two stages run in a CMSSW environment, which the scripts set up themselves from `/cvmfs` if `cmsRun` or `lst_cpu` isn't already on the `PATH`. Only the last stage is Python.

## Layout

- `cmssw/`: shell scripts for the CMSSW stages.
- `samples/<sample>/`: input file lists, one path per line. `relval.txt` lists the step2 files and `trackingntuple.txt` lists the tracking ntuples made from them.
- `src/transformer_datasets/`: the Python package. `t3_processing.py` builds the graphs and `constants.py` holds the ntuple column names.
- `tests/`: unit tests for the Python package.

## Setup

The CMSSW stages need a CMSSW release area with LST built (default: `/ceph/users/atuna/CMSSW_20_1_0_pre2/src`). Override it with `CMSSW_SRC` for `trackingntuple_maker.sh`, or edit `CMSSW_SRC` in `lstntuple_maker.sh`.

For the Python stage, install the package into a virtual environment (Python ≥ 3.10):

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -e .
```

The CPU build of torch is enough; graph building doesn't use a GPU.

## Usage

### 1. Tracking ntuples

Run from the directory where the outputs should go. Each output is named `trackingntuple_<basename of input>`, and existing outputs are skipped.

```bash
mkdir -p qcd_trackingntuple
cd qcd_trackingntuple
while read -r F; do /path/to/transformer-datasets/cmssw/trackingntuple_maker.sh "${F}"; done < /path/to/transformer-datasets/samples/qcd/relval.txt
```

Optional environment variables: `NEVENTS` (default: all), `NTHREADS` (default: 8), `PILEUP_INPUT` (must match the pileup the step2 files were mixed with).

### 2. LST ntuples

`lstntuple_parallel.sh` runs `lstntuple_maker.sh` over every file listed in `lstntuple_maker.txt` in the current directory, at most `MAXJOBS` (default: 10) at a time. Outputs are `LSTNtuple_<name>.root`, with the LST log in `LSTNtuple_<name>.txt`.

```bash
mkdir -p qcd_lstntuple
cd qcd_lstntuple
cp /path/to/transformer-datasets/samples/qcd/trackingntuple.txt lstntuple_maker.txt
/path/to/transformer-datasets/cmssw/lstntuple_parallel.sh
```

### 3. Graphs

```bash
python -m transformer_datasets.t3_processing --input qcd_lstntuple/LSTNtuple_*.root --output qcd_graphs/ --seed 42 -n 16
```

Run with `--help` for all options (`--n_events`, `--split`, `--nofakes`, `--overwrite`, ...).

Each `graph_<N>.pt` is a `torch_geometric.data.Data` with no edges:

| Attribute      | Shape          | Content |
| -------------- | -------------- | ------- |
| `x`            | `[N_T3, 49]`   | T3 features: 3 T3 (pT, η, φ), 2 × 5 LS, 4 × 9 MD (see `constants.py`) |
| `sim_index`    | `[N_T3]`       | matched sim-track index, negative for fake/unmatched |
| `sim_features` | `[N_sim, 8]`   | sim-track features (`SIM_VARS`) |
| `md_layer`     | `[N_T3, 4]`    | layer of each MD |
| `md_simIdx`    | `[N_T3, 4]`    | sim index of each MD |
| `hit_layer`    | `[N_T3, 6]`    | layer of each hit |
| `hit_simIdx`   | `[N_T3, 6]`    | sim index of each hit, resolved to one per hit (`UNMATCHED` = -999) |

transformer-oc depends on the column order of `x`: if you add, remove or reorder features, update the normalization and the η/φ column indices there too.

## Tests

```bash
pytest
```
