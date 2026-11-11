# The test suite

Every test runs on the frozen brain. Two background conditions: **resting brain** = all sensory
neurons and all visual-projection neurons fire randomly at 10 Hz (our stand-in for a living fly's
background); **silent brain** = no background input. Each brain is scored in its own best
condition: the static v2 model's silent brain counts wherever a test can run there.

Vocabulary: *EPG neurons* — the 45 compass neurons arranged around the ellipsoid-body ring; the
active group ("the bump") encodes heading. *PEN neurons* — shift the bump when the fly turns.
*Kim kick* — Kim et al. 2017 protocol: six neighbouring EPGs (one 45° sector) driven at 150 Hz ×
250 synapses per spike for 1.047 s, then darkness. *Kenyon cells* — the ~5,000 neurons of the
mushroom body; in a fly, ~5–10 % respond to any one odour. *Giant fibre* — the escape neuron.
*MN9* — proboscis motor neuron. Rate = spikes per second; R = population-vector localisation
(1 = all activity at one angle); alive = fraction of time bins with R > 0.5.

## Ring attractor (heading memory)

| # | test | protocol | readout | pass |
|---|---|---|---|---|
| 1 | Kim bump, silent brain | kick at 0.5 s, darkness to 4 s | EPGs active after light-off, their rate, width (FWHM), R, alive | persistent, localised, 10–160 Hz |
| 2 | Kim bump, resting brain | same, background on, 3 seeds | same | same |
| 3 | 20 s persistence | kick, darkness to 20 s, 5 seeds | rate / width / angle per 2 s window; bump-neuron rate; ISI CV | alive ≥ 0.75 on every seed, in band, no climb to the ceiling |
| 4 | spontaneous bump | resting brain, no stimulus, 8 s | bump from 1 s on | localised, < 100 Hz |
| 5 | PEN rotation | bump, then left (or right) PEN_a neurons driven for 1 s at 50 syn/spike; decoded angle every 100 ms | endpoint shift and the trajectory (unwrapped rotation, largest 100 ms step) | smooth shifts in opposite directions |
| 6 | second cue | bump at A; a second 45° sector 90° or 180° away driven at 3–4 s (50 or 250 syn/spike) | angle before vs after, bump count | ends within 30° of the cue, one bump |
| 7 | competition | two sectors 180° apart kicked together | bumps after light-off | exactly one, all seeds |
| 8 | drive invariance | one kick at 50 / 100 / 250 / 500 syn/spike | rate and width after light-off | independent of drive |
| 9 | bump width | FWHM of the post-kick profile | degrees | 60–120° (Kim: 90°) |
| 10 | LFT coupling structure | 45-EPG raster 1.6–6 s binned at 2.2 ms; maximum-entropy (Ising) couplings A = −(C + λI)⁻¹; A vs angular distance in 22.5° bins | coupling profile, temporal kernel | positive between neighbours, ≤ 0 across the ring |

## Feeding (silent brain; taste neurons driven for 1 s)

| # | test | pass |
|---|---|---|
| 11 | sugar → MN9 | > 5 Hz |
| 12 | bitter alone | MN9 < 2 Hz |
| 13 | bitter suppresses sugar | MN9 < 70 % of its sugar-alone rate |
| 14 | water → MN9 | > 5 Hz |
| 15 | post-sugar activity decays | neurons still active 1 s after the stimulus < 30 % of those recruited |

## Vision and escape (silent brain)

| # | test | pass |
|---|---|---|
| 16 | looming: LC4 + LPLC2 volley (100 syn/spike, 1 s) | giant fibre > 5 Hz; MDN, DNp09, DNa02 silent |
| 17 | single LC types (LC4, LPLC2, LC6, LC16, LC10a alone, 25 syn/spike) | giant fibre silent |
| 18 | looming evoked activity decays | < 30 % of the recruited surplus after 1 s |

## Olfaction (silent brain; one glomerulus's receptor neurons driven for 1 s)

| # | test | pass |
|---|---|---|
| 19 | DA1 at 3 and 8 syn/spike | 1–15 % of Kenyon cells active |
| 20 | two odours (DA1, then DL3) | each 1–20 %; Jaccard overlap of the two KC sets < 0.5 |
| 21 | concentration series (DA1 at 1 / 3 / 8 / 16) | KC fraction rises monotonically, < 25 % at the top |
| 22 | odour evoked activity decays | < 30 % of the surplus after 1 s |

## Resting state (resting brain, no stimulus, 4 s, non-sensory neurons)

| # | test | pass |
|---|---|---|
| 23 | resting state | sparse; active neurons 1–10 Hz; < 1 % above 200 Hz; ISI CV ~1 |
| 24 | rate distribution | log-rate skew ≥ 0 (a long tail of fast neurons, not a pile at the ceiling) |

## Ablation protocol

One component removed at a time, everything else at the best setting. Readouts: test 3 with
3 seeds, plus rest, olfaction DA1 ×3 and looming. `ablate_check.py` passes an ablation if the
bump is 8–45 Hz and alive ≥ 0.75 on all seeds, rest < 12 Hz, KC 1–15 %, GF > 5 Hz; a passing
ablation gets the full suite.

## Running

```bash
bash run_suite.sh full                 # scorecard (seed 1) + kimrest seeds 2-3 + capability remainder + LFT, best brain
bash run_suite.sh batch "p37a v2" all  # capability battery, both brains (persist/jump/compete/invariance/rest)
bash run_suite.sh suite2               # spontaneous bump, two odours, concentration, evoked decay, both brains
bash run_suite.sh ablate               # the seven ablations (E B C D F G K)
bash run_suite.sh pen "--std 0.2,0.2,recip_ee" "100 101"   # PEN rotation with trajectories
```

`DEVICE=cpp_standalone bash run_suite.sh ...` runs on CPU. Every run appends its result line to
`cap.log` and writes a JSON (`results_cap/cap_<brain>_<test>[_cond]_s<seed>[_flags].json` or
`scorecard_<tag>.json`).
