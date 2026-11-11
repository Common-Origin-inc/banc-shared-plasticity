#!/usr/bin/env bash
# Protocol 16 runner: static brain-and-cord network, bump test, seeds 1 2 3 and 100. Exit codes to manifest.tsv.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
DEV=${DEV:-cuda_standalone}
mkdir -p "$HERE/logs" "$HERE/results"
MAN=$HERE/manifest.tsv; printf 'start\tend\texit\tlog\tcommand\n' > "$MAN"
{ echo "dev=$DEV"; sha256sum "$HERE/../../BANC_BEST_BRAIN/banc_full.npz" "$HERE/../../BANC_BEST_BRAIN/banc_cx_circuit.npz";
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader;
  pip list 2>/dev/null | grep -i -E '^(brian2|brian2cuda|numpy) '; } > "$HERE/environment.txt" 2>&1
for s in 1 2 3 100; do
  t0=$(date -u +%FT%TZ)
  python "$HERE/run_bump.py" --seed $s --device $DEV > "$HERE/logs/bump_s$s.log" 2>&1; rc=$?
  printf '%s\t%s\t%s\t%s\t%s\n' "$t0" "$(date -u +%FT%TZ)" "$rc" "bump_s$s.log" "run_bump.py --seed $s --device $DEV" >> "$MAN"
  tail -1 "$HERE/logs/bump_s$s.log"
done
