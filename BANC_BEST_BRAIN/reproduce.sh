#!/usr/bin/env bash
# Reproduce the best brain from the connectome, end to end, and re-run every test on it.
#
#   bash reproduce.sh [DEVICE] [STAGE]
#     DEVICE  cuda_standalone (default; NVIDIA GPU with brian2cuda)  |  cpp_standalone (CPU, slow)  |  runtime
#     STAGE   all (default) | data | p8 | p37a | score | suite
#
#   data   tools/build_banc_full.py -> banc_full.npz ; tools/build_da_edges.py -> da_edges.npz
#          (needs the BANC bulk tables in $BANC_BULK; both outputs are committed, so optional)
#   p8     100 s homeostatic settle from the v2 weights            -> w_final_p8.npz    (~5 min GPU)
#   p37a   500 s settle with dopamine-gated STDP, warm start p8     -> w_final_p37a.npz  (~25 min GPU)
#   score  the 10-experiment scorecard on the frozen brain, depression on (Kim seeds 1-3, PEN, feeding,
#          looming + single LC types, olfaction x4, rest)                                  (~20 min GPU)
#   suite  capability battery (both brains), additional tests, LFT fingerprint, ablations (~3 h GPU)
#
# Every settle flag below is also recorded inside the saved weights (`params`).
set -e
cd "$(dirname "$0")"
DEV=${1:-cuda_standalone}; STAGE=${2:-all}
PY=${PY:-python}
STD="--std 0.2,0.2,recip_ee"

if [ "$STAGE" = all ] || [ "$STAGE" = data ]; then
  $PY tools/build_banc_full.py
  $PY tools/build_da_edges.py
fi

if [ "$STAGE" = all ] || [ "$STAGE" = p8 ]; then
  # stage 1: Vogels I->E (target 30 Hz, bound 10x) + rate-cap E decay; spontaneous drive incl. visual
  # projection neurons; Kim kick every 10 s; tau_e 15 ms
  $PY run_native_plasticity.py --tag p8 --seconds 100 --tau-e 15 --vis-sens --bump-every 10 --rho0 30 --device "$DEV"
fi

if [ "$STAGE" = all ] || [ "$STAGE" = p37a ]; then
  # stage 2: warm start p8; escape noise 1 mV; iSTDP target 25 Hz, bound 20x; dopamine-gated pair STDP
  # (eta 8e-3, LTD/LTP 0.5, bound 3.5x, gate threshold 0.35, dopamine transient 0.3 s); no E brake;
  # Kim kick every 3 s; 500 s; settle seed 11
  $PY run_native_plasticity.py --tag p37a --seconds 500 --seed 11 --warm w_final_p8.npz \
      --da da_edges.npz --tau-da 0.3 --da-th 0.35 --escape 1 --tau-e 15 --vis-sens --bump-every 3 \
      --rho0 25 --no-e-decay --wmax-i 20 --enable-hebbian-e 8e-3 --ltd-ratio 0.5 --stdp-start 0 \
      --wmax-e 3.5 --device "$DEV"
  $PY gains_by_type.py w_final_p37a.npz
fi

if [ "$STAGE" = all ] || [ "$STAGE" = score ]; then
  $PY validate_scorecard.py --gains w_final_p37a.npz --escape 1 --tau-e 15 --vis $STD \
      --only kim,kimrest,pen,feeding,looming,loom1,olfaction,olf4,rest --seed 1 --device "$DEV"
  for s in 2 3; do
    $PY validate_scorecard.py --gains w_final_p37a.npz --escape 1 --tau-e 15 --vis $STD --only kimrest --seed $s --device "$DEV"
  done
fi

if [ "$STAGE" = all ] || [ "$STAGE" = suite ]; then
  export DEVICE="$DEV"
  waitfor() { until grep -q "$1" cap.log 2>/dev/null; do sleep 30; done; }   # run_suite.sh backgrounds each batch
  bash run_suite.sh full 0.2,0.2,recip_ee stdrecip_eeU0.2t0.2; waitfor "FULL 0.2,0.2,recip_ee DONE"   # scorecard remainder + capability + LFT
  bash run_suite.sh batch "p37a v2" all;                       waitfor "BATCH DONE"                  # capability battery, both brains (p37a = frozen, no STD)
  bash run_suite.sh suite2;                                    waitfor "SUITE2 DONE"                 # spontaneous bump, two odours, concentration, evoked decay
  bash run_suite.sh ablate;                                    waitfor "ABLATE DONE"                 # the seven ablations
  echo "suite done; results in cap.log, results_cap/, results_lft/, scorecard_*.json"
fi
