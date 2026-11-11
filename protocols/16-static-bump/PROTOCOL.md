# Protocol 16: the bump in the static brain-and-cord network

Written before any run (2026-10-16).

## Question

Section 2.1 shows that the wiring supports a persistent bump before any learning. Does the static
76,683-neuron brain-and-cord network, with the same conductance formulation and balance rule as the
learned networks, hold a bump after a localized cue, where on the ring, and at what rate?

## Conditions

As in the 851-neuron extraction: static weights (count-weighted conductances, G = 30 pS, per-neuron
balance rule ρ = 1.8), 5-ms synaptic decay, no background input, no stochastic spiking, no
short-term depression. Cue: the six EPG neurons of the training sector at 150 Hz with events of
250 contacts, 0.5 to 1.547 s; trial 4 s. Seeds 1, 2 and 3, plus seed 100, which repeats the bump
test of the static network's archived scorecard
(`protocols/04-static-rerun/out/results/scorecard_v2init_full_seed1.json`, `kim`) as a check.

## Outcomes (no pass mark)

Per seed: active EPG neurons after the cue (mean rate above 2 Hz from 1.597 to 4 s), their mean
rate, full width at half maximum, population-vector magnitude, persistence (time until the overlap
with the active set at cue offset falls below 0.5, capped at 2.453 s), and which ring positions are
active compared with the stimulated ones.

## Command

`bash protocols/16-static-bump/run.sh` on one GPU. Outputs: `protocols/16-static-bump/results/`.
