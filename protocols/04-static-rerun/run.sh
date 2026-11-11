#!/usr/bin/env bash
# Protocol 04 runner: static reference on banc_full.npz, historical and matched-dynamics
# conditions. Sequential, no retries, every exit status logged; a failed command does not
# stop the run. Static kimrest uses --vis (the proposal in PROTOCOL.md).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
BB=$(cd "$HERE/../../BANC_BEST_BRAIN" && pwd)
PART=${PART:-all}          # PART=rest: only the matched-dynamics capability tests and the two LFT runs, into out_rerun
OUT=$HERE/out; [ "$PART" = rest ] && OUT=$HERE/out_rerun; LOGS=$OUT/logs; RES=$OUT/results
mkdir -p "$LOGS" "$RES"
MAN=$OUT/manifest.tsv
DEV=${DEV:-cuda_standalone}; ALL=kim,kimrest,pen,feeding,looming,loom1,olfaction,olf4,rest
cd "$BB"
printf 'start\tend\texit\tlog\tcommand\n' > "$MAN"
{ git -C "$BB" rev-parse HEAD; git -C "$BB" status --porcelain --untracked-files=no;
  sha256sum banc_full.npz banc_cx_circuit.npz; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader;
  pip list 2>/dev/null | grep -i -E '^(brian2|brian2cuda|numpy|pandas|cython|sympy) '; } > "$OUT/environment.txt" 2>&1

run() {  # run LOGNAME command...
  local log=$LOGS/$1.log; shift
  local t0; t0=$(date -u +%FT%TZ)
  "$@" > "$log" 2>&1; local rc=$?
  printf '%s\t%s\t%s\t%s\t%s\n' "$t0" "$(date -u +%FT%TZ)" "$rc" "$(basename "$log")" "$*" >> "$MAN"
  echo "[$(date -u +%T)] rc=$rc $(basename "$log")"
}

score() {  # score NAME TAG flags...   (the scorecard file name depends only on the flags, so copy after every call)
  local name=$1 tag=$2; shift 2
  local S="python validate_scorecard.py --gains none --vis --device $DEV $*"
  run ${name}_score $S --only $ALL --seed 1;            cp scorecard_$tag.json "$RES/scorecard_${tag}_full_seed1.json" 2>/dev/null
  for s in 2 3; do run ${name}_kimrest_s$s $S --only kimrest --seed $s; cp scorecard_$tag.json "$RES/scorecard_${tag}_kimrest_seed$s.json" 2>/dev/null; done
  run ${name}_pen101 $S --only pen --pen-seed 101;      cp scorecard_$tag.json "$RES/scorecard_${tag}_pen_seed101.json" 2>/dev/null
}
if [ "$PART" = all ]; then
score static  v2init
score matched v2init_stdrecip_eeU0.2t0.2 --tau-e 15 --escape 1 --std 0.2,0.2,recip_ee
fi

cap() {  # cap NAME brain-flags...
  local n=$1; shift
  local C="python capability_tests.py $* --device $DEV"
  for s in 1 2 3 4 5; do run ${n}_persist_s$s $C --test persist --seed $s --seconds 20; done
  for s in 1 2 3; do run ${n}_persist_silent_s$s $C --test persist --seed $s --seconds 20 --silent; done
  for o in 90 180; do for f in 50 250; do run ${n}_jump_o${o}_f$f $C --test jump --offset $o --f $f --seed 1; done; done
  for o in 90 180; do run ${n}_jump_silent_o$o $C --test jump --offset $o --f 250 --seed 1 --silent; done
  for s in 1 2; do run ${n}_compete_s$s $C --test compete --seed $s; run ${n}_spont_s$s $C --test spont --seed $s; done
  run ${n}_spont_silent $C --test spont --seed 1 --silent
  for f in 50 100 250 500; do run ${n}_invariance_f$f $C --test invariance --f $f --seed 1; done
  run ${n}_rest $C --test rest --seed 1
  run ${n}_odors_silent $C --test odors --seed 1 --silent; run ${n}_odors $C --test odors --seed 1
  for f in 1 3 8 16; do run ${n}_conc_f$f $C --test conc --f $f --seed 1 --silent; done
  run ${n}_evoked_silent $C --test evoked --seed 1 --silent; run ${n}_evoked $C --test evoked --seed 1
}
[ "$PART" = all ] && cap static  --brain v2
cap matched --brain p37a --gains none --std 0.2,0.2,recip_ee

# lft_bump.py defaults to tau_e 15 ms, so 5 must be passed for the historical condition
run static_lft  python lft_bump.py --tag v2_rerun   --gains none --tau-e 5 --seconds 6 --device $DEV
run matched_lft python lft_bump.py --tag v2_matched --gains none --escape 1 --tau-e 15 --seconds 6 --std 0.2,0.2,recip_ee --device $DEV

cp -f results_cap/cap_v2_*.json results_cap/cap_p37a_*_gnone*.json "$RES/" 2>/dev/null
cp -f results_lft/lft_v2_rerun.* results_lft/lft_v2_matched.* "$RES/" 2>/dev/null
(cd "$RES" && sha256sum * > "$OUT/sha256.txt")
echo ALL_DONE > "$OUT/progress"
