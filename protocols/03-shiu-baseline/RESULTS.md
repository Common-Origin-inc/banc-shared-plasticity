# Protocol 03 results

The Shiu et al. 2024 formulation (voltage-jump synapses, w_syn 0.275 mV, 1.8 ms delay, uniform
leak) on `banc_full.npz` with this project's sign assignment, stimulated as in the published
code (`out_vstim/`, 54 commands, all exit 0). This is that formulation on
this graph and battery. It is not a score of the published model, which uses the FlyWire graph.

| readout | with the reset of g (published form) | without the reset |
|---|---|---|
| EPG active after the kick, silent / with background | 0 / 0 | 0 / 0 |
| 20 s persistence, alive | 0.00 | 0.00 |
| sugar, MN9 (Hz) | 6.3 | 14.7 |
| bitter / water, MN9 (Hz) | 0.0 / 0.0 | 0.0 / 0.0 |
| sugar + bitter, MN9 (Hz) | 0.0 | 2.1 |
| neurons active during / after sugar | 196 / 0 | 2,109 / 2,055 |
| giant fibre, LC4 + LPLC2 (Hz) | 115.3 | 213.2 |
| giant fibre, LC4 / LPLC2 / LC6 alone (Hz) | 50.5 / 101.1 / 42.6 | 16.8 / 235.3 / 101.1 |
| Kenyon cells active, DA1 (%) | 56.9 | 80.4 |
| two-odor Jaccard | 0.97 | 0.99 |
| neurons active during / after odor | 6,094 / 5,572 | 9,498 / 8,433 |
| rest: mean rate (Hz), ISI CV | 6.2, 0.07 | 17.3, 0.08 |

Stimulated populations fire in this mode (DA1 receptor neurons 113 Hz), unlike in the first run.
In this mode f has no effect, so the DA1 f3 and f8 entries are the same run.

## Reading

- No heading activity outlasts the stimulus, with or without the reset of g. On this graph
  the reset is therefore not what prevents persistence in the voltage-jump formulation; this
  agrees with the two-by-two ablation of the extracted circuit, which attributes persistence
  to conductance synapses.
- The feeding logic is present with the reset: sugar drives MN9, bitter does not, bitter
  suppresses sugar, and activity ends with the stimulus. Water does not drive MN9.
- Odor input recruits more than half of the Kenyon cells and the activity does not end.
