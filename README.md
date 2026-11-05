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

The CPU build of torch is enough. Graph building doesn't use a GPU.

## Usage

### 1. Tracking ntuples

Run from the directory where the outputs should go. For example:

```bash
# voms-proxy-init -voms cms, if necessary
cd samples/qcd/
for F in $(cat relval.txt); do ../../cmssw/trackingntuple_maker.sh $F; done
```

### 2. LST ntuples

`lstntuple_parallel.sh` runs `lstntuple_maker.sh` over every file listed in `trackingntuple.txt` in the current directory. Outputs are `LSTNtuple_<name>.root`. For example:

```bash
cd samples/qcd/
../../cmssw/lstntuple_parallel.sh
```

### 3. Graphs

```bash
python -m transformer_datasets.t3_processing --input qcd_lstntuple/LSTNtuple_*.root --output qcd_graphs/ --seed 42 -n 16
```

Run with `--help` for all options (`--n_events`, `--split`, `--nofakes`, `--overwrite`, ...).

Each `graph_<N>.pt` is a `torch_geometric.data.Data`.

transformer-oc depends on the column order of `x`: if you add, remove or reorder features, update the normalization and the η/φ column indices there too.

## Tests

```bash
pytest
```
