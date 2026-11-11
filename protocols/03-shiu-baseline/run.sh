#!/usr/bin/env bash
# Protocol 03 runner, part A: the Shiu et al. 2024 formulation on banc_full.npz, with the g reset
# (--model shiu, the published form) and without it (--model shiu-noreset). Sequential, no retries,
# every exit status logged; a failed command does not stop the run. Commands are the list in
# PROTOCOL.md with its --synapse/--wsyn/--g-reset flags replaced by --model.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
BB=$(cd "$HERE/../../BANC_BEST_BRAIN" && pwd)
OUT=$HERE/out${OUTSFX:-}; LOGS=$OUT/logs; RES=$OUT/results
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

for M in ${MODES:-shiu shiu-noreset}; do   # MODES="shiu-vstim shiu-vstim-noreset": published stimulation, into out_vstim
  # the scorecard file name depends only on the flags, so remove it before and copy it after every call
  tag=v2init_$M
  S="python validate_scorecard.py --gains none --model $M --vis --device $DEV"
  rm -f scorecard_$tag.json
  run ${M}_score $S --only $ALL --seed 1;   cp scorecard_$tag.json "$RES/scorecard_${tag}_full_seed1.json" 2>/dev/null
  for s in 2 3; do
    rm -f scorecard_$tag.json
    run ${M}_kimrest_s$s $S --only kimrest --seed $s; cp scorecard_$tag.json "$RES/scorecard_${tag}_kimrest_seed$s.json" 2>/dev/null
  done

  C="python capability_tests.py --brain v2 --model $M --device $DEV"
  for s in 1 2 3 4 5; do run ${M}_persist_s$s $C --test persist --seed $s --seconds 20; done
  run ${M}_rest $C --test rest --seed 1
  for o in 90 180; do for f in 50 250; do run ${M}_jump_o${o}_f$f $C --test jump --offset $o --f $f --seed 1; done; done
  for s in 1 2; do run ${M}_compete_s$s $C --test compete --seed $s; run ${M}_spont_s$s $C --test spont --seed $s; done
  for f in 50 100 250 500; do run ${M}_invariance_f$f $C --test invariance --f $f --seed 1; done
  run ${M}_odors_silent $C --test odors --seed 1 --silent; run ${M}_evoked_silent $C --test evoked --seed 1 --silent
  for f in 1 3 8 16; do run ${M}_conc_f$f $C --test conc --f $f --seed 1 --silent; done
  cp -f results_cap/cap_v2_*_$M.json "$RES/" 2>/dev/null
done

(cd "$RES" && sha256sum * > "$OUT/sha256.txt")
echo ALL_DONE > "$OUT/progress"
