#!/usr/bin/env bash
#
# Convert one tracking ntuple into one LSTNtuple, processing all events.
#
# Usage: ./lstntuple_maker.sh TRACKINGNTUPLE
#   TRACKINGNTUPLE: path to a local tracking ntuple, e.g. trackingntuple_<name>.root
#   Output is written to the current directory as LSTNtuple_<name>.root
# Safe to rerun: inputs whose output already exists are skipped.
#
# Run everything in a loop like:
# for F in $(cat trackingntuple.txt); do ./lstntuple_maker.sh "${F}"; done

set -eo pipefail

LST=lst_cpu
START_DIR=${PWD}
CMSSW_SRC=/ceph/users/atuna/CMSSW_20_1_0_pre2/src

if ! command -v "${LST}" > /dev/null; then
    echo "[env ] ${LST} not found, setting up from ${CMSSW_SRC}"
    source /cvmfs/cms.cern.ch/cmsset_default.sh || true
    cd "${CMSSW_SRC}"
    eval "$(scramv1 runtime -sh)" || true # what the `cmsenv` alias does
    source RecoTracker/LSTCore/standalone/setup.sh || true
    cd "${START_DIR}"
fi
if ! command -v "${LST}" > /dev/null; then
    echo "${LST} still not found after environment setup" >&2
    exit 1
fi
if [ -z "${1:-}" ]; then
    echo "Need to provide TRACKINGNTUPLE"
    exit 1
fi

set -euo pipefail

TRACKINGNTUPLE=$(realpath "${1}")

LST_FLAGS=(--md --ls --t3 --t3dnn --t4 --t5 -v 1)

BASENAME=$(basename "${TRACKINGNTUPLE}" .root)
NAME=LSTNtuple_${BASENAME#trackingntuple_}
LSTNTUPLE=${NAME}.root
PARTIAL=${NAME}.partial.root
LOG=${NAME}.txt

if [[ -f "${LSTNTUPLE}" ]]; then
    echo "Skipping ${LSTNTUPLE}, already exists"
    exit 0
fi

echo "Converting ${TRACKINGNTUPLE} into ${LSTNTUPLE}..."

# Write to a .partial file and rename on success, so an interrupted
# job never leaves behind something that looks like a finished output.
${LST} \
    -i ${TRACKINGNTUPLE} \
    -o ${PARTIAL} \
    "${LST_FLAGS[@]}" \
    &> ${LOG}

mv ${PARTIAL} ${LSTNTUPLE}

