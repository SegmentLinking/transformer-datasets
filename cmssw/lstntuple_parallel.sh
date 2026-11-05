#!/usr/bin/env bash
#
# Run lstntuple_maker.sh locally over every file in lstntuple_maker.txt,
# with at most MAXJOBS running in the background at once.
#
# Usage: run from a sample directory containing lstntuple_maker.txt, e.g.
#   cd ttbar_lstntuple && ../lstntuple_parallel.sh
#   MAXJOBS: optional environment variable (default: 10)

set -euo pipefail

MAXJOBS=${MAXJOBS:-10}

# Run in the current (sample) directory; find lstntuple_maker.sh next to this script
MAKER="$(dirname "$(realpath "${BASH_SOURCE[0]}")")/lstntuple_maker.sh"

while read -r FILE; do
    [[ -z "${FILE}" ]] && continue
    while (( $(jobs -rp | wc -l) >= MAXJOBS )); do
        wait -n || true # a failed job should not stop the others
    done
    echo "[launch] ${FILE}"
    "${MAKER}" "${FILE}" &
done < lstntuple_maker.txt

wait
echo "[done] all jobs finished"
