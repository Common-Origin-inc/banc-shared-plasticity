# Protocol 13: removing only the four ER4m sources

Written before any run (2026-10-02).

## Question

The first-token table of protocol 02 removed ten sources: the four ER4m neurons and six
others (FB7B, FB6H, LHPV5e3) that make no contact onto EPG. The faint bump in that arm is
attributed to ER4m by that argument only. Does removing the four ER4m sources alone, with
the other 376 sources kept, give the same outcome?

## Change

Modulation table `da_edges_noer4m.npz`, built by `protocols/01-dopamine/rebuild_without_er4m.py`:
376 sources, 32,374 edges, 63,326 contacts, 2,218 contacts onto EPG (the same EPG total as the
first-token table). Everything else as in protocol 02: same seeds, commands and evaluation.

## Commands

`bash protocols/02-replication/run.sh noer4m K` for K = 1 to 5, one pod each. Outputs:
`out_noer4m/` on each pod, collected to `protocols/13-er4m-only/k<K>/`.

## Outcomes

The protocol 02 primary outcomes per replicate, next to the committed-table and first-token
replicates with the same seeds. No pass mark.

## What would count as informative

If the bump is as faint as in the first-token arm, the dependency is on ER4m. If it is as
strong as in the committed arm, the six other sources mattered and the attribution to ER4m
is wrong.
