# Protocol 04: static reference on the brain-and-cord graph

Written before any run. No result is claimed here.

## Question

What does the static reference score on `banc_full.npz` on every scorecard and capability
test, so that both columns of the results tables refer to one network? Today its feeding and
looming entries and the 5/24 tally come from the central-brain-only graph (49,914 neurons,
used in development and not included here).

## What exists

- `results/capability/cap_v2_*.json` (33 files): the static capability battery was already
  run on `banc_full.npz` (`capability_tests.py` has no `--data` flag and always loads it).
- No static scorecard on `banc_full.npz` is archived. `results/scorecards/` has no
  `scorecard_v2init.json`.
- Matched-dynamics static (ablation E): only `scorecard_ablate_E.json` (rest, olfaction,
  looming), persistence seeds 1 to 3 and one invariance run exist. The full suite was skipped
  because `ablate_check.py` failed it.

## Conditions

1. Historical static: `--gains none`, no escape noise, tau_e 5 ms (no `--tau-e`), no depression.
   In `capability_tests.py` this is `--brain v2`.
2. Matched-dynamics static: `--gains none --tau-e 15 --escape 1 --std 0.2,0.2,recip_ee`.
   In `capability_tests.py` this is `--brain p37a --gains none --std 0.2,0.2,recip_ee`
   (files get the suffix `_gnone`).

## Inputs

`BANC_BEST_BRAIN/banc_full.npz`, `banc_cx_circuit.npz`. The BANC metadata table
`banc_888_meta.feather` (see the README) is required by the feeding test. No checkpoint is read (`--brain p37a` loads `w_final_p37a.npz` and then discards
it when `--gains none` is given).

## Code changes required

1. `validate_scorecard.py` line 444: `BRAIN` is undefined, so `--only feeding` raises NameError
   as committed (see protocol 02, change 1). This protocol cannot produce its main number without it.
2. No other change. Every flag below exists.

## Exact commands

From `BANC_BEST_BRAIN/`, sequentially. `P=../protocols/04-static-rerun`. The scorecard file name
depends only on the flags, so it is copied after every call.

```bash
DEV=cuda_standalone; ALL=kim,kimrest,pen,feeding,looming,loom1,olfaction,olf4,rest
# condition 1, scorecard -> scorecard_v2init.json
S1="python validate_scorecard.py --gains none --vis --device $DEV"
$S1 --only $ALL --seed 1 2>&1 | tee $P/logs/static_score.log
cp scorecard_v2init.json $P/results/scorecard_v2init_full_seed1.json
for s in 2 3; do $S1 --only kimrest --seed $s; cp scorecard_v2init.json $P/results/scorecard_v2init_kimrest_seed$s.json; done
$S1 --only pen --pen-seed 101; cp scorecard_v2init.json $P/results/scorecard_v2init_pen_seed101.json
# condition 2, scorecard -> scorecard_v2init_stdrecip_eeU0.2t0.2.json
S2="$S1 --tau-e 15 --escape 1 --std 0.2,0.2,recip_ee"; T2=v2init_stdrecip_eeU0.2t0.2
$S2 --only $ALL --seed 1 2>&1 | tee $P/logs/matched_score.log
cp scorecard_$T2.json $P/results/scorecard_${T2}_full_seed1.json
for s in 2 3; do $S2 --only kimrest --seed $s; cp scorecard_$T2.json $P/results/scorecard_${T2}_kimrest_seed$s.json; done
$S2 --only pen --pen-seed 101; cp scorecard_$T2.json $P/results/scorecard_${T2}_pen_seed101.json
# capability battery, both conditions
for B in "--brain v2" "--brain p37a --gains none --std 0.2,0.2,recip_ee"; do
  C="python capability_tests.py $B --device $DEV"
  for s in 1 2 3 4 5; do $C --test persist --seed $s --seconds 20; done
  for s in 1 2 3; do $C --test persist --seed $s --seconds 20 --silent; done
  for o in 90 180; do for f in 50 250; do $C --test jump --offset $o --f $f --seed 1; done; done
  for o in 90 180; do $C --test jump --offset $o --f 250 --seed 1 --silent; done
  for s in 1 2; do $C --test compete --seed $s; $C --test spont --seed $s; done
  $C --test spont --seed 1 --silent
  for f in 50 100 250 500; do $C --test invariance --f $f --seed 1; done
  $C --test rest --seed 1
  $C --test odors --seed 1 --silent; $C --test odors --seed 1
  for f in 1 3 8 16; do $C --test conc --f $f --seed 1 --silent; done
  $C --test evoked --seed 1 --silent; $C --test evoked --seed 1
done 2>&1 | tee $P/logs/capability.log
# LFT fingerprint (lft_bump.py defaults to tau_e 15 ms, so 5 must be passed for condition 1)
python lft_bump.py --tag v2_rerun --gains none --tau-e 5 --seconds 6 --device $DEV
python lft_bump.py --tag v2_matched --gains none --escape 1 --tau-e 15 --seconds 6 --std 0.2,0.2,recip_ee --device $DEV
```

The capability lines are those of `run_suite.sh batch`, `silent` and `suite2`, written out
because `suite2` hard-codes the pair of brains. Persistence seeds 4 and 5 are added for the
static conditions so that they match the learned network's five.

Points fixed by the code that a reader should know:
- `--vis` changes only `kimrest`. The scorecard `rest` test always uses a sensory-only
  background, for every brain. `capability_tests.py` uses sensory plus visual projection
  neurons unless `--no-vis`.
- The shorter list `--only feeding,looming,olf4` is not enough. It omits kim, kimrest, pen,
  loom1, olfaction and rest, which the static column also needs from this graph.

## Seeds

kim 100 (hard-coded); kimrest `--seed` 1, 2, 3; PEN `--pen-seed` 100 (default) and 101;
feeding, looming, olfaction, olf4: 1 (the `sim()` default); scorecard rest 3 (hard-coded);
capability and LFT as given on each line.

## Outputs

Written by the code into `BANC_BEST_BRAIN/` (`scorecard_*.json`, `results_cap/`, `results_lft/`),
then copied to `protocols/04-static-rerun/results/` with `manifest.json` (commit hash, command,
SHA-256 of inputs and outputs). The archived `results/` directory is not touched.

## Outcomes

Primary, for each of the two conditions: the eight continuous readouts of protocol 02, from the
same JSON fields, plus the remaining static entries of the summary table (median
active rate, active count, KC at 16 synapses, two-odor Jaccard, neurons active 1 s after odor).
Secondary: the 24-criterion tally for each condition, under the numeric definitions fixed in
protocol 02, change 4.
Check: the 30 archived `cap_v2_*.json` that this protocol repeats (the three `noorn` files are
not repeated) are compared field by field with the reruns. Any difference is recorded with both values.

## Decision rule

- The rerun values replace the central-brain values in both tables, whatever they are,
  including the sugar entry. The tally is recomputed and the new count
  replaces 5/24 even if it is the same.
- `TESTS.md` says the static model's silent-brain result counts "wherever a test can run
  there". Proposal: the tally uses the same background
  condition per test as the learned column, and the silent-brain static values are reported beside it.
- Static `kimrest` uses `--vis` (the background the `--brain v2` preset already uses; this is
  what `run.sh` runs). The alternative, sensory-only, would be `S1` without `--vis`.
- Condition 2 is reported as the matched comparison for learning. Condition 1 stays as the historical comparison.
- If an archived capability value does not reproduce, the rerun value is used and the difference is noted.

## Estimated GPU time (earlier timings, RTX 4090)

Scorecard about 20 min per condition with the extra kimrest seeds (`reproduce.sh`), plus two
PEN programs. Capability: 32 programs per condition at about 1 min each (40 to 90 s compile,
README). Two LFT runs. About 1 h per condition, 2 GPU h in total.

## Dependencies

Independent of protocol 01 (no modulation, no learning). Needs
the BANC metadata table `banc_888_meta.feather` (see the README) and code change 1. Its
outputs are the static column that protocols 02 and 03 are compared against.
