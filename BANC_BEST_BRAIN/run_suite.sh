#!/usr/bin/env bash
# test-suite runner (GPU by default; DEVICE=cpp_standalone or runtime to override): cap_run.sh one "<flags>"        -> one capability run, filtered output
#                cap_run.sh batch "<brains>" [what] -> whole battery in the background, log cap.log
#                cap_run.sh silent "<brains>"      -> silent-brain persistence (3 seeds) + two-cue jumps
cd "$(dirname "$0")"
DEVICE=${DEVICE:-cuda_standalone}
[ -d /usr/local/cuda/bin ] && export PATH=/usr/local/cuda/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
[ -f /workspace/venv/bin/activate ] && source /workspace/venv/bin/activate
one() {
  python capability_tests.py $1 --device $DEVICE 2>&1 | grep -aE '^(\[|CAP)|Error|error:|Traceback|File "' | cut -c1-600
}
gr() { grep -aE --line-buffered '^(KIM|REST|OLF|LOOM|FEED|PEN|         \||    ring|Kim|gains)|Error|error:|Traceback' | cut -c1-420; }
case "$1" in
  one) one "$2" ;;
  ablate)    # overnight: E first, then B C D F G K; full suite on every ablation that passes ablate_check.py
    sp=0.2,0.2,recip_ee; base="stdrecip_eeU0.2t0.2"; G37="--gains w_final_p37a.npz"
    abl() {  # NAME capflags scflags capsuffix
      echo "== ablation $1 ($(date))"
      for s in 1 2 3; do one "--brain p37a --test persist --seed $s --seconds 20 --std $sp $2"; done
      python validate_scorecard.py --only rest,olfaction,looming --std $sp $3 --device $DEVICE 2>&1 | gr
      new=$(ls -t scorecard_*.json | head -1); cp "$new" "scorecard_ablate_$1.json"
      if python ablate_check.py "$1" "$4" "scorecard_ablate_$1.json"; then
        echo "== full suite for ablation $1 ($(date))"
        python validate_scorecard.py --only kim,kimrest,pen,feeding,loom1,olf4 --std $sp $3 --seed 1 --device $DEVICE 2>&1 | gr
        new=$(ls -t scorecard_*.json | head -1); cp "$new" "scorecard_ablate_$1_full.json"
        for s in 2 3; do python validate_scorecard.py --only kimrest --std $sp $3 --seed $s --device $DEVICE 2>&1 | gr; done
        for o in 90 180; do for f in 50 250; do one "--brain p37a --test jump --offset $o --f $f --seed 1 --std $sp $2"; done; done
        for s in 1 2; do one "--brain p37a --test compete --seed $s --std $sp $2"; done
        for f in 50 100 250 500; do one "--brain p37a --test invariance --f $f --seed 1 --std $sp $2"; done
        one "--brain p37a --test rest --seed 1 --std $sp $2"
        for s in 4 5; do one "--brain p37a --test persist --seed $s --seconds 20 --std $sp $2"; done
      fi
    }
    which=${2:-"E B C D F G K"}; waitfor=${3:-none}   # cap_run.sh ablate "F G K" "GATED DONE"
    (
    if [ "$waitfor" != none ]; then until grep -q "$waitfor" cap.log; do sleep 30; done; fi
    for a in $which; do case $a in
      E) abl E "--gains none"           "--gains none --vis --tau-e 15 --escape 1"            "${base}_gnone" ;;
      B) abl B "--no-escape"            "$G37 --vis --tau-e 15"                              "${base}_noesc" ;;
      C) abl C "--tau-e 5"              "$G37 --vis --tau-e 5 --escape 1"                    "${base}_taue5" ;;
      D) abl D "--gains w_final_p8.npz" "--gains w_final_p8.npz --vis --tau-e 15 --escape 1" "${base}_gp8" ;;
      F) abl F "--gains-class ie"       "$G37 --gains-class ie --vis --tau-e 15 --escape 1"  "${base}_gie" ;;
      G) abl G "--gains-class ee"       "$G37 --gains-class ee --vis --tau-e 15 --escape 1"  "${base}_gee" ;;
      K) abl K "--no-vis"               "$G37 --tau-e 15 --escape 1"                         "${base}_novis" ;;
    esac; done
    echo "ABLATE DONE $(date)"
    ) >> cap.log 2>&1 &
    echo "ablation batch started (pid $!)" ;;
  gated)     # reciprocity-gated brake in the settle: p8r (100 s) -> scorecard -> p37ar (500 s from p8r) -> scorecards +/- STD
    sp=0.2,0.2,recip_ee; waitfor=${2:-none}          # cap_run.sh gated [ABLATE DONE|none]
    (
    if [ "$waitfor" != none ]; then until grep -q "$waitfor" cap.log; do sleep 30; done; fi
    echo "== gated p8r ($(date))"
    python run_native_plasticity.py --tag p8r --seconds 100 --tau-e 15 --vis-sens --bump-every 10 --rho0 30 --brake-recip --device $DEVICE 2>&1 \
      | grep -aE 'warm|brake|saved|final|Error|error:|Traceback' | cut -c1-300
    python gains_by_type.py w_final_p8r.npz 2>&1 | cut -c1-300
    echo "== p8r scorecard (feeding first)"
    python validate_scorecard.py --gains w_final_p8r.npz --tau-e 15 --vis --only feeding,rest,olfaction,looming,kimrest --seed 1 --device $DEVICE 2>&1 | gr
    echo "== gated p37ar ($(date))"
    python run_native_plasticity.py --tag p37ar --seconds 500 --seed 11 --warm w_final_p8r.npz --da da_edges.npz --tau-da 0.3 --da-th 0.35 \
      --escape 1 --tau-e 15 --vis-sens --bump-every 3 --rho0 25 --no-e-decay --wmax-i 20 --enable-hebbian-e 8e-3 --ltd-ratio 0.5 \
      --stdp-start 0 --wmax-e 3.5 --device $DEVICE 2>&1 | grep -aE 'warm|saved|final|Error|error:|Traceback' | cut -c1-300
    python gains_by_type.py w_final_p37ar.npz 2>&1 | cut -c1-300
    echo "== p37ar frozen scorecard (no STD)"
    python validate_scorecard.py --gains w_final_p37ar.npz --escape 1 --tau-e 15 --vis --only kim,kimrest,pen,feeding,looming,loom1,olfaction,olf4,rest --seed 1 --device $DEVICE 2>&1 | gr
    echo "== p37ar + STD scorecard"
    python validate_scorecard.py --gains w_final_p37ar.npz --escape 1 --tau-e 15 --vis --std $sp --only kim,kimrest,pen,feeding,looming,loom1,olfaction,olf4,rest --seed 1 --device $DEVICE 2>&1 | gr
    for s in 2 3; do python validate_scorecard.py --gains w_final_p37ar.npz --escape 1 --tau-e 15 --vis --std $sp --only kimrest --seed $s --device $DEVICE 2>&1 | gr; done
    echo "== p37ar + STD capability"
    for s in 1 2 3; do one "--brain p37a --gains w_final_p37ar.npz --test persist --seed $s --seconds 20 --std $sp"; done
    for s in 1 2 3; do one "--brain p37a --gains w_final_p37ar.npz --test persist --seed $s --seconds 20"; done
    for o in 90 180; do one "--brain p37a --gains w_final_p37ar.npz --test jump --offset $o --f 250 --seed 1 --std $sp"; done
    one "--brain p37a --gains w_final_p37ar.npz --test compete --seed 1 --std $sp"
    for f in 50 500; do one "--brain p37a --gains w_final_p37ar.npz --test invariance --f $f --seed 1 --std $sp"; done
    one "--brain p37a --gains w_final_p37ar.npz --test rest --seed 1 --std $sp"
    echo "GATED DONE $(date)"
    ) >> cap.log 2>&1 &
    echo "gated pipeline queued behind the ablations (pid $!)" ;;
  batch)
    brains=${2:-"p37a v2"}; what=${3:-all}
    (
    for b in $brains; do
      if [ "$what" = all ] || [ "$what" = persist ]; then for s in 1 2 3; do one "--brain $b --test persist --seed $s --seconds 20"; done; fi
      if [ "$what" = all ] || [ "$what" = jump ]; then for o in 90 180; do for f in 50 250; do one "--brain $b --test jump --offset $o --f $f --seed 1"; done; done; fi
      if [ "$what" = all ] || [ "$what" = compete ]; then for s in 1 2; do one "--brain $b --test compete --seed $s"; done; fi
      if [ "$what" = all ] || [ "$what" = invariance ]; then for f in 50 100 250 500; do one "--brain $b --test invariance --f $f --seed 1"; done; fi
      if [ "$what" = all ] || [ "$what" = rest ]; then one "--brain $b --test rest --seed 1"; fi
    done
    echo "BATCH DONE $(date)"
    ) >> cap.log 2>&1 &
    echo "batch started (pid $!)" ;;
  silent)
    brains=${2:-"p37a v2"}
    (
    for b in $brains; do
      for s in 1 2 3; do one "--brain $b --test persist --seed $s --seconds 20 --silent"; done
      for o in 90 180; do one "--brain $b --test jump --offset $o --f 250 --seed 1 --silent"; done
    done
    echo "SILENT DONE $(date)"
    ) >> cap.log 2>&1 &
    echo "silent batch started (pid $!)" ;;
  live)      # option 1: iSTDP on during a 20 s persistence test, p37a, seeds given
    seeds=${2:-"1 2 3"}
    (
    for s in $seeds; do one "--brain p37a --test persist --seed $s --seconds 20 --live"; done
    echo "LIVE DONE $(date)"
    ) >> cap.log 2>&1 &
    echo "live batch started (pid $!)" ;;
  live2)     # option 1 continued: second cue, competition, drive invariance with iSTDP on
    (
    for o in 90 180; do one "--brain p37a --test jump --offset $o --f 250 --seed 1 --seconds 8 --live"; done
    one "--brain p37a --test compete --seed 1 --seconds 8 --live"
    for f in 50 500; do one "--brain p37a --test invariance --f $f --seed 1 --seconds 8 --live"; done
    echo "LIVE2 DONE $(date)"
    ) >> cap.log 2>&1 &
    echo "live2 batch started (pid $!)" ;;
  live_eta)  # option 1, slower learning: cap_run.sh live_eta "1e-3 3e-4" "1 2 3"
    etas=${2:-"1e-3"}; seeds=${3:-"1 2 3"}
    (
    for e in $etas; do for s in $seeds; do one "--brain p37a --test persist --seed $s --seconds 20 --live --eta-i $e"; done; done
    echo "LIVE_ETA DONE $(date)"
    ) >> cap.log 2>&1 &
    echo "live_eta batch started (pid $!)" ;;
  controls)  # what makes the live bump hop: frozen + 3 s pre-settle, and live with eta_i 0
    (
    for s in 1 2; do one "--brain p37a --test persist --seed $s --seconds 20 --pre 3"; done
    for s in 1 2; do one "--brain p37a --test persist --seed $s --seconds 20 --live --eta-i 0"; done
    echo "CONTROLS DONE $(date)"
    ) >> cap.log 2>&1 &
    echo "controls batch started (pid $!)" ;;
  std)       # option 2: short-term depression on excitatory synapses. cap_run.sh std "0.2,0.2 0.2,0.5 0.5,0.2 0.5,0.5" "1"
    specs=${2:-"0.2,0.2 0.2,0.5 0.5,0.2 0.5,0.5"}; seeds=${3:-"1"}
    (
    for sp in $specs; do for s in $seeds; do one "--brain p37a --test persist --seed $s --seconds 20 --std $sp"; done; done
    echo "STD DONE $(date)"
    ) >> cap.log 2>&1 &
    echo "std batch started (pid $!)" ;;
  std2)      # option 2 follow-up at one setting: more seeds, neighbours, and the other tests. cap_run.sh std2 0.2,0.2
    sp=${2:-0.2,0.2}
    (
    for s in 2 3; do one "--brain p37a --test persist --seed $s --seconds 20 --std $sp"; done
    for nb in 0.1,0.2 0.2,0.1 0.3,0.2; do one "--brain p37a --test persist --seed 1 --seconds 20 --std $nb"; done
    one "--brain p37a --test rest --seed 1 --std $sp"
    for o in 90 180; do one "--brain p37a --test jump --offset $o --f 250 --seed 1 --std $sp"; done
    one "--brain p37a --test compete --seed 1 --std $sp"
    for f in 50 500; do one "--brain p37a --test invariance --f $f --seed 1 --std $sp"; done
    echo "STD2 DONE $(date)"
    ) >> cap.log 2>&1 &
    echo "std2 batch started (pid $!)" ;;
  score)     # frozen scorecard (rest, olfaction, looming, feeding, kimrest) with extra flags. cap_run.sh score "--std 0.2,0.2"
    flags=${2:-}
    (
    python validate_scorecard.py --gains w_final_p37a.npz --only rest,olfaction,looming,feeding,kimrest --vis --tau-e 15 --escape 1 $flags --device $DEVICE 2>&1 \
      | grep -aE '^(KIM|REST|OLF|LOOM|FEED|PEN|         \||    ring|short-term|escape)|Error|error:|Traceback' | cut -c1-400
    echo "SCORE DONE $(date)"
    ) >> cap.log 2>&1 &
    echo "score started (pid $!)" ;;
  livestd)   # option 3: iSTDP live + STD. cap_run.sh livestd "0.2,0.2" "1 2 3" [eta]
    sp=${2:-0.2,0.2}; seeds=${3:-"1 2 3"}; eta=${4:-}
    (
    for s in $seeds; do one "--brain p37a --test persist --seed $s --seconds 20 --live --std $sp ${eta:+--eta-i $eta}"; done
    echo "LIVESTD DONE $(date)"
    ) >> cap.log 2>&1 &
    echo "livestd batch started (pid $!)" ;;
  opt3)      # plan: mechanism run (STD 0.2,0.2 + live) then propagation scorecards at weak STD, sequentially
    (
    for s in 1 2 3; do one "--brain p37a --test persist --seed $s --seconds 20 --live --std 0.2,0.2"; done
    echo "LIVESTD DONE $(date)"
    for sp in 0.1,0.2 0.2,0.1 0.1,0.1; do
      python validate_scorecard.py --gains w_final_p37a.npz --only rest,olfaction,looming,feeding,kimrest --vis --tau-e 15 --escape 1 --std $sp --device $DEVICE 2>&1 \
        | grep -aE '^(KIM|REST|OLF|LOOM|FEED|PEN|         \||    ring|short-term)|Error|error:|Traceback' | cut -c1-400
      echo "SCORE $sp DONE $(date)"
    done
    echo "OPT3 DONE $(date)"
    ) >> cap.log 2>&1 &
    echo "opt3 batch started (pid $!)" ;;
  smoke)     # 3 s GPU build+run with the given flags; prints the CAP line or the error
    python capability_tests.py --brain p37a --test invariance --f 250 --seed 1 --seconds 3 $2 --device $DEVICE 2>&1 \
      | grep -aE --line-buffered '^(\[|CAP)|Error|error:|Traceback|File "' | cut -c1-600 ;;
  recip)     # reciprocity-scaled depression battery: cap_run.sh recip "0.4,0.2,recip"; aborts at the first Traceback
    sp=${2:-0.4,0.2,recip}
    (
    ok=1
    for s in 1 2 3; do
      out=$(python capability_tests.py --brain p37a --test persist --seed $s --seconds 20 --std $sp --device $DEVICE 2>&1 | grep -aE '^(\[|CAP)|Error|error:|Traceback|File "' | cut -c1-600)
      echo "$out"; echo "$out" | grep -q Traceback && { echo "ABORT: error in persist seed $s"; ok=0; break; }
    done
    if [ $ok = 1 ]; then
      python validate_scorecard.py --gains w_final_p37a.npz --only rest,olfaction,looming,feeding,kimrest --vis --tau-e 15 --escape 1 --std $sp --device $DEVICE 2>&1 \
        | grep -aE --line-buffered '^(KIM|REST|OLF|LOOM|FEED|PEN|         \||    ring|short-term)|Error|error:|Traceback' | cut -c1-400
      for o in 90 180; do one "--brain p37a --test jump --offset $o --f 250 --seed 1 --std $sp"; done
      one "--brain p37a --test compete --seed 1 --std $sp"
      for f in 50 500; do one "--brain p37a --test invariance --f $f --seed 1 --std $sp"; done
    fi
    echo "RECIP $sp DONE $(date)"
    ) >> cap.log 2>&1 &
    echo "recip batch started (pid $!)" ;;
  pos)       # position tests at one setting, after any running python finishes: cap_run.sh pos "0.2,0.2,recip_ee"
    sp=${2:-0.2,0.2,recip_ee}
    (
    while pgrep -x python > /dev/null; do sleep 10; done
    for o in 90 180; do one "--brain p37a --test jump --offset $o --f 250 --seed 1 --std $sp"; done
    for s in 1 2; do one "--brain p37a --test compete --seed $s --std $sp"; done
    for f in 50 500; do one "--brain p37a --test invariance --f $f --seed 1 --std $sp"; done
    for s in 4 5; do one "--brain p37a --test persist --seed $s --seconds 20 --std $sp"; done
    echo "POS $sp DONE $(date)"
    ) >> cap.log 2>&1 &
    echo "pos batch queued (pid $!)" ;;
  full)      # full suite at one setting: cap_run.sh full "0.2,0.2,recip_ee" TAG
    sp=${2:-0.2,0.2,recip_ee}; tg=${3:-stdrecip_eeU0.2t0.2}
    SC="python validate_scorecard.py --gains w_final_p37a.npz --vis --tau-e 15 --escape 1 --std $sp --device $DEVICE"
    gr() { grep -aE --line-buffered '^(KIM|REST|OLF|LOOM|FEED|PEN|         \||    ring|Kim)|Error|error:|Traceback' | cut -c1-420; }
    (
    echo "== scorecard full (seed 1)"
    $SC --only kim,kimrest,pen,feeding,looming,loom1,olfaction,olf4,rest --seed 1 2>&1 | gr
    cp scorecard_p37a_$tg.json scorecard_p37a_${tg}_full_seed1.json
    for s in 2 3; do
      echo "== kimrest seed $s"
      $SC --only kimrest --seed $s 2>&1 | gr
      cp scorecard_p37a_$tg.json scorecard_p37a_${tg}_kimrest_seed$s.json
    done
    echo "== capability remainder"
    for o in 90 180; do one "--brain p37a --test jump --offset $o --f 50 --seed 1 --std $sp"; done
    for f in 100 250; do one "--brain p37a --test invariance --f $f --seed 1 --std $sp"; done
    one "--brain p37a --test rest --seed 1 --std $sp"
    for s in 1 2 3; do one "--brain p37a --test persist --seed $s --seconds 20 --silent --std $sp"; done
    echo "== LFT"
    python lft_bump.py --tag p37a_$tg --gains w_final_p37a.npz --escape 1 --tau-e 15 --seconds 6 --std $sp --device $DEVICE 2>&1 \
      | grep -aE '^(\[|LFT)|Error|error:|Traceback' | cut -c1-400
    echo "FULL $sp DONE $(date)"
    ) >> cap.log 2>&1 &
    echo "full suite started (pid $!)" ;;
  suite2)    # additional tests, best brain vs v2 (silent brain = v2's published condition; resting brain too)
    BEST="--brain p37a --std 0.2,0.2,recip_ee"
    (
    for b in "$BEST" "--brain v2"; do
      for s in 1 2; do one "$b --test spont --seed $s"; done
      one "$b --test spont --seed 1 --silent"
      one "$b --test odors --seed 1 --silent"; one "$b --test odors --seed 1"
      for f in 1 3 8 16; do one "$b --test conc --f $f --seed 1 --silent"; done
      one "$b --test evoked --seed 1 --silent"; one "$b --test evoked --seed 1"
    done
    echo "SUITE2 DONE $(date)"
    ) >> cap.log 2>&1 &
    echo "suite2 started (pid $!)" ;;
  noorn)     # resting brain without ORNs in the spontaneous set: is the MB sparse then?
    BEST="--brain p37a --std 0.2,0.2,recip_ee"
    (
    for b in "$BEST" "--brain v2"; do
      one "$b --test odors --seed 1 --no-orn"; one "$b --test rest --seed 1 --no-orn"; one "$b --test spont --seed 1 --no-orn"
    done
    one "$BEST --test persist --seed 1 --seconds 20 --no-orn"
    one "$BEST --test evoked --seed 1 --no-orn"
    echo "NOORN DONE $(date)"
    ) >> cap.log 2>&1 &
    echo "noorn started (pid $!)" ;;
  pen)       # PEN rotation test (scorecard 'pen') with extra flags, foreground: cap_run.sh pen "--std 0.2,0.2,recip_ee" [seeds]
    for s in ${3:-100}; do
      echo "pen seed $s"
      python validate_scorecard.py --gains w_final_p37a.npz --only pen --vis --tau-e 15 --escape 1 $2 --pen-seed $s --device $DEVICE 2>&1 \
        | grep -aE --line-buffered '^(PEN|KIM)|         \||    ring|Error|error:|Traceback' | cut -c1-400
    done ;;
esac
