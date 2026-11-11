# Protocol 16 results

Static brain-and-cord network, no background input, 5-ms synapses; cue: six EPG neurons of the training
sector, 0.5 to 1.547 s, 150 Hz, 250 contacts per event; 4-s trials. Measured values, no threshold.

| seed | active EPG after the cue | their rate (Hz) | width (deg) | R | persistence (s) | stimulated neurons active | bump angle (deg) | cue angle (deg) |
|---|---|---|---|---|---|---|---|---|
| 1 | 13 | 288.0 | 96 | 0.891 | 2.453 (window 2.453) | 6 of 6 | -177 | -154 |
| 2 | 14 | 268.3 | 96 | 0.890 | 2.453 (window 2.453) | 6 of 6 | -177 | -154 |
| 3 | 13 | 288.1 | 96 | 0.891 | 2.453 (window 2.453) | 6 of 6 | -177 | -154 |
| 100 | 13 | 289.9 | 96 | 0.890 | 2.453 (window 2.453) | 6 of 6 | -177 | -154 |

Seed 100 repeats the bump test of the archived scorecard of the same network
(`protocols/04-static-rerun/out/results/scorecard_v2init_full_seed1.json`, `kim`: 13 EPG, 289.87 Hz, 96 deg) and agrees.
Persistence equals the whole post-cue window in every seed: the active set at the end of the trial is the set active at cue offset.
The bump angle is the rate-weighted circular mean over the post-cue window; the active set contains all six stimulated neurons.
