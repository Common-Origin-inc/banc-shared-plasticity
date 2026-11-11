#!/usr/bin/env bash
# Protocol 02 runner. One arm per invocation, strictly sequential (the settle build
# directory is fixed, so two settles must never run at once on one machine).
#
# usage: run.sh ARM [K...]      ARM = sub   -> da_edges.npz        (committed table)
#                               ARM = first -> da_edges_first.npz
#        K defaults to 1 2 3 4 5.  Arm "sub" also runs step 0 (seed 7 stage 1).
#        ROT=N in the environment (protocol 07): training kicks cycle through N sectors in both
#        stages (--bump-rotate N); tags get the suffix r<N>, outputs go to out_<ARM>_rot<N>, no step 0.
#
# Stage 1 does not use the modulation table, so both arms use the same stage-1 seeds and
# should produce identical rep<k>_p8 checkpoints; the manifest hashes let that be checked.
# Nothing is retried. Every command's exit status goes to manifest.tsv, and a failed
# command does not stop the run.
set -u
ARM=$1; shift
KS=${*:-1 2 3 4 5}
case $ARM in
  sub)   DA=da_edges.npz;       SFX="" ;;
  first) DA=da_edges_first.npz; SFX="f" ;;
  noer4m) DA=da_edges_noer4m.npz; SFX="e" ;;   # protocol 13: only the four ER4m sources removed
  nolpsp) DA=da_edges_nolpsp.npz; SFX="l" ;;   # protocol 15: only the two LPsP sources removed
  *) echo "ARM must be sub, first, noer4m or nolpsp"; exit 2 ;;
esac
ROT=${ROT:-0}; ROTF=""; OSFX=""
if [ "$ROT" -gt 0 ]; then ROTF="--bump-rotate $ROT"; SFX="${SFX}r${ROT}"; OSFX="_rot$ROT"; fi
# X2="flags" XTAG=name: extra stage-2 flags; tags get the suffix XTAG, outputs go to out_<ARM>_<XTAG>, no step 0
X2=${X2:-}; XTAG=${XTAG:-}
if [ -n "$XTAG" ]; then SFX="${SFX}${XTAG}"; OSFX="${OSFX}_$XTAG"; fi
HERE=$(cd "$(dirname "$0")" && pwd)
BB=$(cd "$HERE/../../BANC_BEST_BRAIN" && pwd)
OUT=$HERE/out_$ARM$OSFX; LOGS=$OUT/logs; RES=$OUT/results
mkdir -p "$LOGS" "$RES"
MAN=$OUT/manifest.tsv
DEV=${DEV:-cuda_standalone}; STD=0.2,0.2,recip_ee
cd "$BB"
printf 'start\tend\texit\tlog\tcommand\n' > "$MAN"
{ echo "arm=$ARM da=$DA ks=$KS dev=$DEV"; git -C "$BB" rev-parse HEAD; git -C "$BB" status --porcelain --untracked-files=no;
  sha256sum banc_full.npz banc_cx_circuit.npz "$DA" w_final_p8.npz w_final_p37a.npz; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader;
  pip list 2>/dev/null | grep -i -E '^(brian2|brian2cuda|numpy|pandas|cython|sympy) '; } > "$OUT/environment.txt" 2>&1

run() {  # run LOGNAME command...
  local log=$LOGS/$1.log; shift
  local t0; t0=$(date -u +%FT%TZ)
  "$@" > "$log" 2>&1; local rc=$?
  printf '%s\t%s\t%s\t%s\t%s\n' "$t0" "$(date -u +%FT%TZ)" "$rc" "$(basename "$log")" "$*" >> "$MAN"
  echo "[$(date -u +%T)] rc=$rc $(basename "$log")"
}

if [ "$ARM" = sub ] && [ "$ROT" = 0 ] && [ -z "$XTAG" ]; then
  run rep0_p8 python run_native_plasticity.py --tag rep0_p8 --seconds 100 --seed 7 --tau-e 15 --vis-sens \
      --bump-every 10 --rho0 30 --device $DEV
fi

for k in $KS; do
  R=rep${k}${SFX}                       # stage-2 and evaluation tag
  P8=rep${k}_p8; [ "$ROT" -gt 0 ] && P8=rep${k}r${ROT}_p8       # stage 1 differs from protocol 02 only when kicks rotate
  run $P8 python run_native_plasticity.py --tag $P8 --seconds 100 --seed $((100+k)) --tau-e 15 \
      --vis-sens --bump-every 10 --rho0 30 $ROTF --device $DEV
  run ${R}_p37a python run_native_plasticity.py --tag ${R}_p37a --seconds 500 --seed $((200+k)) \
      --warm w_final_$P8.npz --da $DA --tau-da 0.3 --da-th 0.35 --escape 1 --tau-e 15 \
      --vis-sens --bump-every 3 --rho0 25 --no-e-decay --wmax-i 20 --enable-hebbian-e 8e-3 \
      --ltd-ratio 0.5 --stdp-start 0 --wmax-e 3.5 $ROTF $X2 --device $DEV
  G=w_final_${R}_p37a.npz; T=${R}_p37a_stdrecip_eeU0.2t0.2
  run ${R}_gains python gains_by_type.py $G
  SC="python validate_scorecard.py --gains $G --escape 1 --tau-e 15 --vis --std $STD --device $DEV"
  run ${R}_score $SC --only kim,kimrest,pen,feeding,looming,loom1,olfaction,olf4,rest --seed 1
  cp scorecard_$T.json "$RES/scorecard_${T}_full_seed1.json" 2>/dev/null   # same name is overwritten below
  for s in 2 3; do
    run ${R}_kimrest_s$s $SC --only kimrest --seed $s
    cp scorecard_$T.json "$RES/scorecard_${T}_kimrest_seed$s.json" 2>/dev/null
  done
  CAP="python capability_tests.py --brain p37a --gains $G --std $STD --device $DEV"
  for s in 1 2 3 4 5; do run ${R}_persist_s$s $CAP --test persist --seed $s --seconds 20; done
  run ${R}_rest $CAP --test rest --seed 1
  run ${R}_ablate_check python ablate_check.py $R stdrecip_eeU0.2t0.2_g${R}_p37a "$RES/scorecard_${T}_full_seed1.json"
  # secondary: the remaining capability tests, same conditions as the archived suite
  for o in 90 180; do for f in 50 250; do run ${R}_jump_o${o}_f$f $CAP --test jump --offset $o --f $f --seed 1; done; done
  for s in 1 2; do run ${R}_compete_s$s $CAP --test compete --seed $s; run ${R}_spont_s$s $CAP --test spont --seed $s; done
  for f in 50 100 250 500; do run ${R}_invariance_f$f $CAP --test invariance --f $f --seed 1; done
  run ${R}_odors $CAP --test odors --seed 1 --silent
  run ${R}_evoked $CAP --test evoked --seed 1 --silent
  for f in 1 3 8 16; do run ${R}_conc_f$f $CAP --test conc --f $f --seed 1 --silent; done
  for s in 1 2 3; do run ${R}_persist_silent_s$s $CAP --test persist --seed $s --seconds 20 --silent; done
  run ${R}_lft python lft_bump.py --tag $T --gains $G --escape 1 --tau-e 15 --seconds 6 --std $STD --device $DEV
  # collect this replicate's outputs
  cp -f w_final_$P8.npz w_final_${R}_p37a.npz log_$P8.json log_${R}_p37a.json "$RES/" 2>/dev/null
  cp -f results_cap/cap_p37a_*_g${R}_p37a*.json "$RES/" 2>/dev/null
  cp -f results_lft/lft_${T}.* "$RES/" 2>/dev/null
  (cd "$RES" && sha256sum * > "$OUT/sha256.txt")
  echo "replicate $k done" > "$OUT/progress"
done
[ "$ARM" = sub ] && [ "$ROT" = 0 ] && [ -z "$XTAG" ] && cp -f w_final_rep0_p8.npz log_rep0_p8.json "$RES/" 2>/dev/null
(cd "$RES" && sha256sum * > "$OUT/sha256.txt")
echo ALL_DONE > "$OUT/progress"
