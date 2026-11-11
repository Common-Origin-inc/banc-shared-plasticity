#!/usr/bin/env bash
# Protocol 14 runner: 20-s persistence at eight cue positions, three seeds, one network.
# usage: run.sh NET     NET = P (w_final_p37a.npz) or K = 1..5 (protocol 02 replicate K)
# Strictly sequential on one machine. Nothing is retried; exit codes go to manifest.tsv.
set -u
NET=$1
HERE=$(cd "$(dirname "$0")" && pwd)
BB=$(cd "$HERE/../../BANC_BEST_BRAIN" && pwd)
OUT=$HERE/$NET; LOGS=$OUT/logs; RES=$OUT/results
mkdir -p "$LOGS" "$RES"
MAN=$OUT/manifest.tsv
DEV=${DEV:-cuda_standalone}; STD=0.2,0.2,recip_ee
cd "$BB"
printf 'start\tend\texit\tlog\tcommand\n' > "$MAN"

run() {  # run LOGNAME command...
  local log=$LOGS/$1.log; shift
  local t0; t0=$(date -u +%FT%TZ)
  "$@" > "$log" 2>&1; local rc=$?
  printf '%s\t%s\t%s\t%s\t%s\n' "$t0" "$(date -u +%FT%TZ)" "$rc" "$(basename "$log")" "$*" >> "$MAN"
  echo "[$(date -u +%T)] rc=$rc $(basename "$log")"
}

if [ "$NET" = P ]; then GF=""; G=""; PAT="cap_p37a_persist*_s?_std*eeU0.2t0.2.json"
else
  # Replicate checkpoints are not in the repository tree (release weights-v1). If absent, retrain replicate NET
  # with the protocol 02 stage-1 and stage-2 commands and seeds, and compare against the archived hash.
  G=w_final_rep${NET}_p37a.npz
  if [ ! -f "$G" ]; then
    run rep${NET}_p8 python run_native_plasticity.py --tag rep${NET}_p8 --seconds 100 --seed $((100+NET)) --tau-e 15 \
        --vis-sens --bump-every 10 --rho0 30 --device $DEV
    run rep${NET}_p37a python run_native_plasticity.py --tag rep${NET}_p37a --seconds 500 --seed $((200+NET)) \
        --warm w_final_rep${NET}_p8.npz --da da_edges.npz --tau-da 0.3 --da-th 0.35 --escape 1 --tau-e 15 \
        --vis-sens --bump-every 3 --rho0 25 --no-e-decay --wmax-i 20 --enable-hebbian-e 8e-3 \
        --ltd-ratio 0.5 --stdp-start 0 --wmax-e 3.5 --device $DEV
    cp -f w_final_rep${NET}_p8.npz w_final_rep${NET}_p37a.npz log_rep${NET}_p8.json log_rep${NET}_p37a.json "$RES/" 2>/dev/null
  fi
  GF="--gains $G"; PAT="cap_p37a_persist*_grep${NET}_p37a.json"
fi
{ echo "net=$NET dev=$DEV"; git -C "$BB" rev-parse HEAD; git -C "$BB" status --porcelain --untracked-files=no;
  sha256sum banc_full.npz banc_cx_circuit.npz da_edges.npz w_final_p37a.npz $G; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader;
  pip list 2>/dev/null | grep -i -E '^(brian2|brian2cuda|numpy|pandas|cython|sympy) '; } > "$OUT/environment.txt" 2>&1

CAP="python capability_tests.py --brain p37a $GF --std $STD --device $DEV"
for c in 0 45 90 135 180 225 270 315; do
  for s in 1 2 3; do run persist_c${c}_s$s $CAP --test persist --seed $s --seconds 20 --cue-offset $c; done
done
cp -f results_cap/$PAT "$RES/" 2>/dev/null
(cd "$RES" && sha256sum * > "$OUT/sha256.txt")
echo ALL_DONE > "$OUT/progress"
