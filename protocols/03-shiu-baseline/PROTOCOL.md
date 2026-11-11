# Protocol 03: the published formulation as a baseline

Written before any run. No result is claimed here.

## Question

A. How does the Shiu et al. 2024 formulation (voltage-jump synapse, `w_syn` 0.275 mV per
synapse, uniform leak, reset of v and g to zero) score on the same tests on the same graph
(`banc_full.npz`)?

B. In the 851-neuron central-complex circuit, does resetting g at each spike change the outcome
of the voltage-jump cells of the existing ablation? The archived ablation never applies it.

## What exists in the code

- `heading_circuit_extraction/ring_attractor_sim.py`, `build(voltage_synapse=True)`:
  `dv/dt = (v_0 - v + g)/t_mbr`, `dg/dt = -g/tau_syn`, both `(unless refractory)`, `on_pre: g_post += w`,
  w = count x W_SYN mV (inhibitory counts negative), `reset="v = v_rst"` only, no delay.
  With `uniform_leak=True` the per-neuron scale is 1, so every synapse is exactly W_SYN.
  With `uniform_leak=False` the jump is scaled by median g_L / g_L,i.
- `heading_circuit_extraction/ablation.py`: the 2x2 (synapse x leak), W_SYN 0.275, 0.40, 0.55, 0.70, 1.0 mV,
  seeds 100, 101, 102, 8 EPGs driven 0.5 to 1.0 s, run to 3.5 s. No command-line arguments.
  It always writes `experiment1_ablation.json`, so a rerun overwrites the archived file.
- The module docstring states that the g reset is not used. No code path applies it.
- `BANC_BEST_BRAIN/plasticity.py` `build_network` has one membrane equation (conductance,
  area-scaled leak). There is no voltage-jump mode on the 76,683-neuron graph, and
  `validate_scorecard.py`, `capability_tests.py`, `lft_bump.py` have no flag for one.

## Code changes required

1. `ring_attractor_sim.build`: add `g_reset=False`; when true, the voltage branch uses
   `reset="v = v_rst; g = 0*mV"`. Default keeps archived behaviour. This is a one-line change,
   but the reset string is in `ring_attractor_sim.py`, not in `ablation.py`, and
   `heading_circuit_extraction/` is kept frozen as the archived experiment. The choice between
   editing it and copying both files into this directory is made before the run.
2. `ablation.py` (or a copy `ablation_greset.py`): loop over `g_reset`, write to a new file name.
3. `plasticity.build_network`: a `shiu` mode. State g in volts; the equations above; reset
   of v, and of g when `g_reset`; w = n_syn x sign x `w_syn`; no balance rule, no area, no
   gains, no escape noise, no depression; stimulus and background events add
   f x `w_syn` to g, with the same f the tests already pass (`STIM_HZ` 150, f synapses per event).
4. Flags `--synapse shiu`, `--wsyn 0.275`, `--g-reset 0|1` in `validate_scorecard.py`,
   `capability_tests.py` and `lft_bump.py`, each adding a suffix to the output tag so that
   nothing collides with `scorecard_v2init.json` or `cap_v2_*.json`.
5. `validate_scorecard.py` line 444: `BRAIN` is undefined (see protocol 02, change 1).
6. Synaptic delay: none of the builders has one. See "Not comparable".

None of these flags exists today. The commands in part A are the intended form and cannot run
until changes 3 to 5 are made and reviewed.

## Inputs

`heading_circuit_extraction/banc_cx_circuit.npz` (851 x 851). `BANC_BEST_BRAIN/banc_full.npz`,
`banc_cx_circuit.npz`. The BANC metadata table `banc_888_meta.feather` (see the README) is
needed for the feeding test. No checkpoint is used.

## Exact commands

Part B, CPU, from `heading_circuit_extraction/` (after changes 1 and 2):

```bash
python ablation_greset.py 2>&1 | tee ../protocols/03-shiu-baseline/logs/ablation_greset.log
```

Cells: synapse {voltage, conductance} x leak {uniform, area} x g reset {off, on}. The four
reset-off cells repeat the archived 2x2 and must reproduce `experiment1_ablation.json`. The two
voltage cells with reset on are the new measurement. The two conductance cells with
reset on (g_e and g_i set to zero at a spike) are optional; they are not the published model
of anyone.

Part A, GPU, from `BANC_BEST_BRAIN/` (after changes 3 to 5). `R` is 1 (published form, primary) then 0:

```bash
X="--gains none --synapse shiu --wsyn 0.275 --g-reset $R --device cuda_standalone"
python validate_scorecard.py $X --vis --only kim,kimrest,pen,feeding,looming,loom1,olfaction,olf4,rest --seed 1
for s in 2 3; do python validate_scorecard.py $X --vis --only kimrest --seed $s; done   # copy the JSON after each
C="python capability_tests.py --brain v2 --synapse shiu --wsyn 0.275 --g-reset $R --device cuda_standalone"
for s in 1 2 3 4 5; do $C --test persist --seed $s --seconds 20; done
$C --test rest --seed 1
for o in 90 180; do for f in 50 250; do $C --test jump --offset $o --f $f --seed 1; done; done
for s in 1 2; do $C --test compete --seed $s; $C --test spont --seed $s; done
for f in 50 100 250 500; do $C --test invariance --f $f --seed 1; done
$C --test odors --seed 1 --silent; $C --test evoked --seed 1 --silent
for f in 1 3 8 16; do $C --test conc --f $f --seed 1 --silent; done
```

`w_syn` is fixed at 0.275 mV. No value is tuned on these tests. Tests that need a background
use this repository's convention (10 Hz events of 250 synapses on sensory and visual
projection neurons); the published model has no such background.

## Seeds

Part B: 100, 101, 102 (as archived). Part A: the same evaluation seeds as protocols 02 and 04.

## Outputs

Part B: `protocols/03-shiu-baseline/results/ablation_2x2x2.json` and the log.
Part A: scorecard and capability JSONs (written next to the code, then copied to
`protocols/03-shiu-baseline/results/`), with commit hash, commands and SHA-256 in `manifest.json`.

## Outcomes

Part A primary: the eight continuous readouts defined in protocol 02 (same JSON fields).
Secondary: the 24-criterion tally under the same numeric definitions as protocol 02.
Part B primary: per cell and W_SYN, mean and SD over three seeds of `live_frac`, `R_after`,
`pr_after`, `active_hz_after`, `n_active_after`, `persist_s`. A cell "has a persistent
localised bump" under the rule already in `ablation.py`: `live_frac` > 0.9, `pr_after` < 20, `R_after` > 0.35.

## Decision rule

Part A has no pass or fail. The numbers are reported as a third column beside the static
reference and the learned network, whatever they are, with g reset on as the baseline and
g reset off as a sensitivity row.
Part B: if any voltage cell changes its persistent-localised verdict at any W_SYN between
reset off and on, the paper reports the reset as a separate factor. If none changes, the
paper states that the reset did not alter the verdict over 0.275 to 1.0 mV.
If a reset-off cell fails to reproduce the archived JSON, that is recorded before anything else is interpreted.

## Not comparable

- Graph. This is BANC (brain and nerve cord, most optic-lobe-intrinsic neurons excluded,
  fast synapses of verified modulator neurons dropped, sign from first verified transmitter).
  Shiu et al. used FlyWire with their own sign rule. Scores here are not a replication of their paper.
- What had been verified of their code before this protocol is only: `dv/dt = (v_0 - v + g)/t_mbr`, `dg/dt = -g/tau`,
  reset `v = v_rst; w = 0; g = 0*mV`, `on_pre: g += w`, `w_syn = 0.275 mV`, `r_poi = 150 Hz`.
- An alpha-shaped synapse and a 1.8 ms delay are not recorded anywhere in this repository,
  and the reset of a variable `w` is recorded but not explained. These are checked against
  `model.py` before change 3 is written. Until then the baseline is "the recorded equations,
  no delay" and must be named that way.
- The tests and thresholds were chosen while developing this repository's model.

## Estimated time

Part B: CPU only (Brian2 runtime, 851 neurons, 8 cells x 5 values x 3 seeds = 120 runs of
3.5 s). No earlier timing exists for it. Part A: one scorecard about 20 min and 24 capability
programs at about 1 min each (README and `reproduce.sh` figures), about 45 min per reset
setting, about 1.5 GPU h for both.

## Dependencies

Independent of protocol 01 (no modulation table, no learning). Shares code change 5 and the
tally definitions with protocol 02, and its comparison column with protocol 04.

## Amendments before the first run (2026-10-02)

- The mode is selected with `--model shiu` (published form) or `--model shiu-noreset`
  (same, without the reset of g at a spike); the `--synapse/--wsyn/--g-reset` flags named
  above were not implemented. w_syn is fixed at 0.275 mV. Commands: `run.sh`.
- Confirmed from the published `model.py` and implemented: the 1.8 ms synaptic delay, and
  both equations frozen while refractory.
- Not comparable, in addition to the points above: the published model delivers stimulation
  as a jump on v (w_syn times 250, with no refractory period for stimulated neurons), so every
  input event makes the stimulated neuron fire. Here an event of f synapses adds f times w_syn
  to g, the project's convention, so weakly driven populations (for example taste neurons at
  25 synapse equivalents) may fire less than in the published model. The firing of the
  stimulated populations must be reported with the results before any comparison is drawn.
  A run with the published stimulation convention is a possible follow-up and is not part of
  this protocol.
- The graph is BANC, not FlyWire, with this project's sign assignment. The outcome is the
  score of that formulation on this graph and battery. It is not a score of the published model.
