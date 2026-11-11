# Protocol 02: independent learning runs

Written before any run. No result is claimed here.

## Question

Is the outcome of the trained model a property of the method or of one run? Every
learned-network number so far comes from one stage-1 run and one stage-2 run.

## What the code does with the learning seed

- `run_native_plasticity.py` line 45: `seed_val = int(arg("--seed", 7, int))`, then
  `start_scope(); seed(seed_val)` (Brian2 seed, line 70). This is the only learning seed.
- `reproduce.sh` stage `p8` passes no `--seed`, so stage 1 used 7. Stage `p37a` passes `--seed 11`.
- The seed is not written into the checkpoint. `params` in `w_final_p8.npz` and
  `w_final_p37a.npz` has no `seed` key (checked). The only record is the command line.
- Build directory is fixed (`/tmp/brian_settle`), so two settles must never run at once on one machine.

## Inputs

All in `BANC_BEST_BRAIN/`: `banc_full.npz`, `banc_cx_circuit.npz`, `da_edges.npz`,
`w_final_p8.npz`, `w_final_p37a.npz`. The scorecard feeding test also reads the BANC metadata
table `banc_888_meta.feather` (sha256 `86ccf5df...780098`, 57,503,026 bytes), which is not
redistributed here (see the README).
Modulation table: `da_edges.npz`, or the rebuilt file if protocol 01 decides Rebuild (`$DA` below).

## Code changes required (before the first run)

1. `validate_scorecard.py`: define the path to the BANC metadata table, which the feeding test
   reads (`--only feeding` raises NameError without it).
2. Record the seed in the checkpoint `params` (adds `seed` and `provenance`); the two published
   checkpoints lack it.
3. The ten new checkpoints are kept out of version control, with their SHA-256 hashes recorded.
4. No script computes the 24-criterion tally, and criteria 8, 10 and 23 in `TESTS.md`
   ("independent of drive", coupling profile sign, "ISI CV ~1") have no numeric threshold in
   code. Numeric definitions must be written and committed before the secondary outcome is scored.

## Seeds

Proposal: replicate k = 1..5 uses stage-1 seed 100+k and stage-2 seed 200+k (none equal to 7
or 11). Evaluation seeds are the same for every replicate: persistence 1 to 5; scorecard
`--seed 1` (kimrest also 2, 3); kim 100 and rest 3 are hard-coded; feeding, looming and
olfaction use the `sim()` default 1; PEN `--pen-seed` default 100.

## Exact commands

Run from `BANC_BEST_BRAIN/`, sequentially. `P=../protocols/02-replication`.

```bash
DEV=cuda_standalone; STD=0.2,0.2,recip_ee; DA=da_edges.npz
# step 0 (proposal): does seed 7 reproduce the published stage 1 on this hardware?
python run_native_plasticity.py --tag rep0_p8 --seconds 100 --seed 7 --tau-e 15 --vis-sens \
    --bump-every 10 --rho0 30 --device $DEV 2>&1 | tee $P/logs/rep0_p8.log
for k in 1 2 3 4 5; do
  python run_native_plasticity.py --tag rep${k}_p8 --seconds 100 --seed $((100+k)) --tau-e 15 \
      --vis-sens --bump-every 10 --rho0 30 --device $DEV 2>&1 | tee $P/logs/rep${k}_p8.log
  python run_native_plasticity.py --tag rep${k}_p37a --seconds 500 --seed $((200+k)) \
      --warm w_final_rep${k}_p8.npz --da $DA --tau-da 0.3 --da-th 0.35 --escape 1 --tau-e 15 \
      --vis-sens --bump-every 3 --rho0 25 --no-e-decay --wmax-i 20 --enable-hebbian-e 8e-3 \
      --ltd-ratio 0.5 --stdp-start 0 --wmax-e 3.5 --device $DEV 2>&1 | tee $P/logs/rep${k}_p37a.log
  python gains_by_type.py w_final_rep${k}_p37a.npz 2>&1 | tee $P/logs/rep${k}_gains.log
  G=w_final_rep${k}_p37a.npz; T=rep${k}_p37a_stdrecip_eeU0.2t0.2
  SC="python validate_scorecard.py --gains $G --escape 1 --tau-e 15 --vis --std $STD --device $DEV"
  $SC --only kim,kimrest,pen,feeding,looming,loom1,olfaction,olf4,rest --seed 1 2>&1 | tee $P/logs/rep${k}_score.log
  cp scorecard_$T.json $P/results/scorecard_${T}_full_seed1.json   # same name is overwritten below
  for s in 2 3; do $SC --only kimrest --seed $s; cp scorecard_$T.json $P/results/scorecard_${T}_kimrest_seed$s.json; done
  CAP="python capability_tests.py --brain p37a --gains $G --std $STD --device $DEV"
  for s in 1 2 3 4 5; do $CAP --test persist --seed $s --seconds 20; done
  $CAP --test rest --seed 1
  python ablate_check.py rep$k stdrecip_eeU0.2t0.2_grep${k}_p37a $P/results/scorecard_${T}_full_seed1.json
  # secondary (24-criterion tally), same conditions as the archived suite
  for o in 90 180; do for f in 50 250; do $CAP --test jump --offset $o --f $f --seed 1; done; done
  for s in 1 2; do $CAP --test compete --seed $s; $CAP --test spont --seed $s; done
  for f in 50 100 250 500; do $CAP --test invariance --f $f --seed 1; done
  $CAP --test odors --seed 1 --silent; $CAP --test evoked --seed 1 --silent
  for f in 1 3 8 16; do $CAP --test conc --f $f --seed 1 --silent; done
  for s in 1 2 3; do $CAP --test persist --seed $s --seconds 20 --silent; done
  python lft_bump.py --tag $T --gains $G --escape 1 --tau-e 15 --seconds 6 --std $STD --device $DEV
done
```

`--brain p37a` only selects the evaluation preset (escape 1 mV, tau_e 15 ms); `--gains` replaces
the weights. `run_suite.sh` is not used because it hard-codes `w_final_p37a.npz`.

## Outputs

The code writes next to itself: `w_final_rep<k>_{p8,p37a}.npz`, `log_rep<k>_*.json`,
`scorecard_*.json`, `results_cap/cap_p37a_*_grep<k>_p37a.json`, `results_lft/`. After each
replicate, copy all of them to `protocols/02-replication/results/` and add to
`manifest.json`: commit hash, both seeds, full commands, SHA-256 of every input and output,
exit status. `w_final_p8.npz` and `w_final_p37a.npz` are never written (tags differ).

## Primary outcomes (continuous, per replicate)

| readout | source field |
|---|---|
| non-driven mean rate | `cap_*_rest_s1_*.json: mean_hz` (driven set excluded, 4 s, sensory + visual projection background) |
| fraction above 200 Hz | same file, `ceiling` |
| window-active EPG rate | `cap_*_persist_s<1..5>_*.json: hz_mean` (mean over 2 s windows of EPGs above 2 Hz) |
| alive fraction | same files, `alive` (bins with R > 0.5, averaged over windows) |
| bump width | same files, `fwhm_mean` |
| KC fraction, DA1 | `scorecard: olfaction.DA1_f3.kc_frac` (3 synapses per spike, silent, seed 1) |
| GF rate | `scorecard: looming["LC4+LPLC2"].gf` |
| MN9 sugar rate | `scorecard: feeding.sugar.mn9` |

Reported per replicate and per evaluation seed, with min, median, max across replicates,
next to the archived values of the published checkpoint. The scorecard `rest.mean` is reported
too but is a different quantity: `--vis` affects only `kimrest`, so scorecard rest uses a
sensory-only background, 2 s, seed 3. The scorecard `alive` is also a different quantity
(`live_frac`, summed EPG rate above 2 Hz).

Secondary: the 24-criterion tally per replicate, under the definitions fixed in code change 4.

## Reporting rule (decided 2026-09-30)

No pass mark is fixed in advance, and no replicate is labelled a success or a failure by a
threshold. Every primary outcome is reported as its measured value for every replicate and
every evaluation seed, next to the published checkpoint, with min, median and max across
replicates. The reader judges the spread.

The existing `ablate_check.py` output and the 24-criterion tally are reported per replicate
as descriptive secondary information, with their historical thresholds stated, and are not
used to include, exclude or relabel any run.

Fixed in advance all the same: the seeds, the commands, the outcome definitions, and the
rule that nothing is dropped. A crashed run, a NaN or a missing file is reported as such and
stays in the tables. No seed is rerun or replaced. More replicates may only be appended
(seeds 106+, 206+) and are declared as such. Step 0 is not a replicate; if its gains differ
from `w_final_p8.npz`, that is recorded.

Both modulation tables are run (protocol 01 decision): the committed `da_edges.npz` and a
rebuilt table keeping only sources whose first verified token is dopamine. Same seeds for both.

## Estimated GPU time (earlier timings, RTX 4090)

Stage 1 about 5 min, stage 2 about 25 min, scorecard about 20 min with the two extra
kimrest seeds (`reproduce.sh`); 28 capability and LFT programs at roughly 1 min each
(40 to 90 s compile per program, README). About 80 min per replicate, 6.5 to 7 h for five, plus 5 min for step 0.

## Dependencies

- Protocol 01: the choice of `$DA` (both tables are run; see below).
- The current uniform 10 Hz background is used, so that the method is tested as it was used.
- The BANC metadata table and code change 1, for feeding.
