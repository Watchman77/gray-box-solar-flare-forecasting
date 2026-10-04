#!/bin/sh
# One CPU-only collection, with an external wall-time bound and no retry.
set -eu
if [ "$#" -ne 3 ]; then
  echo 'Usage: run_aia72_cpu_collection.sh RETURN_SHA256 COLLECTOR_SHA256 LAUNCHER_SHA256' >&2
  exit 2
fi
collection_source_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec /usr/bin/timeout --signal=TERM --kill-after=10s 600s \
  /usr/bin/env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
  /home/abmoses2000/aia_gpu_venv/bin/python -u \
  "$collection_source_dir/archive_aia72_completion_training.py" \
  --return-sha256 "$1" --source-sha256 "$2" --launcher-sha256 "$3"
