#!/bin/bash
# Run every experiment config that does not yet have results (resumable).
cd "$(dirname "$0")/.."
SCRATCH="${SCRATCH_DIR:-/tmp}"
run_one() {
  cfg="$1"
  name="$(basename "$cfg" .json)"
  out="./dynamic_results/exp_${name}/summary.csv"
  if [ -f "$out" ]; then echo "skip $name (done)"; return 0; fi
  echo "start $name"
  TF_CPP_MIN_LOG_LEVEL=3 OMP_NUM_THREADS=1 \
    python dynamic_main.py --mode margin --config "$cfg" \
    > "$SCRATCH/exp_${name}.log" 2>&1 \
    && echo "done $name" || echo "FAILED $name (see $SCRATCH/exp_${name}.log)"
}
export -f run_one
export SCRATCH
ls experiments/*.json | xargs -n1 -P "${PARALLEL:-3}" -I{} bash -c 'run_one "$@"' _ {}
echo ALL-EXPERIMENTS-FINISHED
