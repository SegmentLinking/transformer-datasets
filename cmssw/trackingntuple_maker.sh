#!/usr/bin/env bash
#
# Convert a Phase-2 PU200 step2 (GEN-SIM-DIGI-RAW) file into a tracking ntuple.
#
# Usage: make_trackingntuple.sh STEP2_FILEIN
#   STEP2_FILEIN is a path to a local file (relative or absolute).
#   Output is written to the current directory as trackingntuple_<basename of input>.
#
# Optional environment variables:
#   CMSSW_SRC     CMSSW release area to set up if cmsRun is not available
#   NEVENTS       number of events to process (default: -1, i.e. all)
#   NTHREADS      number of threads (default: 8)
#   PILEUP_INPUT  pileup input for cmsDriver.py (default: the MinBias dataset the
#                 QCD D121 step2 files were mixed with; must match the step2 file!)

if [ -z "${1:-}" ]; then
    echo "Need to provide STEP2_FILEIN" >&2
    exit 1
fi

CMSSW_SRC=${CMSSW_SRC:-/ceph/users/atuna/CMSSW_20_1_0_pre2/src}

if ! command -v cmsRun > /dev/null; then
    echo "[env ] cmsRun not found, setting up from ${CMSSW_SRC}"
    source /cvmfs/cms.cern.ch/cmsset_default.sh
    pushd "${CMSSW_SRC}" > /dev/null
    eval "$(scramv1 runtime -sh)" # aka `cmsenv`
    popd > /dev/null
fi

if ! command -v cmsRun > /dev/null; then
    echo "cmsRun still not found after environment setup" >&2
    exit 1
fi

set -euo pipefail

STEP2_FILEIN=${1}
NEVENTS=${NEVENTS:--1}
NTHREADS=${NTHREADS:-8}
PILEUP_INPUT=${PILEUP_INPUT:-das:/RelValMinBias_14TeV/CMSSW_20_0_0_pre1-150X_mcRun4_realistic_v1_MinBiasGS_D121_GSOnly_noPU-v2/GEN-SIM}

FILEIN=file:$(realpath "${STEP2_FILEIN}")

BASENAME=$(basename "${STEP2_FILEIN}")
FILEOUT=trackingntuple_${BASENAME}
PYFILE=step3_${BASENAME%.root}.py

if [[ -f "${FILEOUT}" ]]; then
    echo "Skipping ${FILEOUT}, already exists"
    exit 0
fi

echo "[info] host     : $(hostname)"
echo "[info] CMSSW    : ${CMSSW_VERSION} (${CMSSW_BASE})"
echo "[info] input    : ${FILEIN}"
echo "[info] output   : $(pwd)/${FILEOUT}"
echo "[info] events   : ${NEVENTS}, threads: ${NTHREADS}"
echo "[info] pileup   : ${PILEUP_INPUT}"

ANCHOR='process.mix.digitizers = cms.PSet()'
NEWTEXT='process.simHitTPAssocProducer.simHitSrc += ["g4SimHits:TrackerHitsPixelBarrelHighTof", "g4SimHits:TrackerHitsPixelEndcapHighTof"]'

cmsDriver.py step3 \
     -s RAW2DIGI,RECO,RECOSIM,PAT,VALIDATION:@phase2Validation+@miniAODValidation,DQM:@phase2+@miniAODDQM \
     -n ${NEVENTS} \
     --conditions auto:phase2_realistic_T35 \
     --datatier GEN-SIM-RECO,MINIAODSIM,DQMIO \
     --eventcontent FEVTDEBUGHLT,MINIAODSIM,DQM \
     --geometry ExtendedRun4D121 \
     --era Phase2C22I13M9 \
     --pileup AVE_200_BX_25ns \
     --pileup_input ${PILEUP_INPUT} \
     --nThreads ${NTHREADS} \
     --lazy_download \
     --customise Validation/RecoTrack/customiseTrackingNtuple.customiseTrackingNtuple \
     --customise Validation/RecoTrack/customiseTrackingNtuple.extendedContent \
     --customise_command="process.trackingNtuple.clusterMasks = []; process.TFileService.fileName = '${FILEOUT}'" \
     --filein ${FILEIN} \
     --fileout file:step3_unused.root \
     --python_filename ${PYFILE} \
     --no_exec
sed -i "s/${ANCHOR}/${ANCHOR}\n${NEWTEXT}/g" ${PYFILE}

if ! grep -qF "${NEWTEXT}" ${PYFILE}; then
    echo "Failed to add HighTof sim hits to ${PYFILE}: anchor '${ANCHOR}' not found" >&2
    exit 1
fi

time cmsRun ${PYFILE}

echo "[done] wrote $(pwd)/${FILEOUT}"
