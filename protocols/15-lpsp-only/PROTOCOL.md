# Protocol 15: removing only the two LPsP sources (control for protocol 13)

Written before any run (2026-10-16).

## Question

Protocol 13 showed that removing the four ER4m sources from the modulation table leaves a
faint bump in four of five replicates. ER4m supplies 2,355 of the 4,573 modulatory contacts
onto EPG neurons (51.5 %). Is the dependence specific to ER4m, or would removing any
comparable share of the gating onto EPG have the same effect?

The matched comparison is LPsP: two sources, 7,965 contacts in the table (ER4m: 7,581),
2,090 onto EPG (45.7 %), and like ER4m they contact all 45 EPG neurons. Unlike ER4m, LPsP
neurons are annotated as modulatory only (`modulator:dopamine`), so they keep no fast
synapses in the model; ER4m also inhibits EPG through fast GABA synapses.

## Change

Modulation table `da_edges_nolpsp.npz`, built by `protocols/01-dopamine/rebuild_without_lpsp.py`:
the committed table without the two LPsP sources. Arm `nolpsp` added to
`protocols/02-replication/run.sh`. Everything else as in protocols 02 and 13: same seeds,
commands and evaluation.

## Commands

`bash protocols/02-replication/run.sh nolpsp K` for K = 1 to 5, one pod each. Outputs:
`out_nolpsp/` on each pod, collected to `protocols/15-lpsp-only/k<K>/`.

## Outcomes

The protocol 02 primary outcomes per replicate (rest, EPG rate, localization, width, giant
fibre), next to the committed-table and ER4m-removed replicates with the same seeds. No pass mark.

## What would count as informative

If the bump stays as strong as with the committed table, removing a similar share of the
modulatory contacts onto EPG is not enough, and the dependence is specific to ER4m (to its
gating together with its firing, which differs from LPsP's). If the bump becomes as faint as
in the ER4m-removed arm, the dependence is on the total gating onto EPG, and the text must
not attribute it to ER4m specifically.
