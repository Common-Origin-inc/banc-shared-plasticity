# Protocol 14: persistence at every cue position

Written before any run (2026-10-16).

## Question

Every persistence test so far cued the bump at one sector, the one the networks were trained
on (the hub sector of `kim_stim`, offset 0). The short paper therefore describes the learned
state as "a stable heading bump at the trained location, tested there". Two things are open:

1. Does a cue at any other position of the ring leave a persistent bump?
2. If it does, does the bump stay where it was cued, or move to a preferred position?

The angles already archived by protocol 02 suggest that the answer differs between runs: in
replicate 5 the bump after the training cue sits about 70 degrees from the cue, and the
spontaneous bump (no cue) sits at the same place.

## Change

`capability_tests.py --cue-offset DEG` rotates the cue sector around the ring by DEG (the
same rotation `--offset` already applies to the second stimulus of the jump test). With the
default 0 the command, the tag and the outputs are unchanged. Results also record
`cue_offset` and `cue_angle` (circular mean of the ring angles of the six stimulated EPG neurons).

## Networks and commands

Frozen networks, no training: network P (`w_final_p37a.npz`) and the five committed-table
replicates of protocol 02 (`out_sub/results/w_final_rep<k>_p37a.npz`). For each network, cue
offsets 0, 45, ..., 315 degrees, evaluation seeds 1 to 3, the 20-s persistence test with
reciprocity-scaled depression, exactly as in protocol 02:

    python capability_tests.py --brain p37a [--gains W] --std 0.2,0.2,recip_ee \
        --device cuda_standalone --test persist --seed S --seconds 20 --cue-offset C

`bash protocols/14-heading-map/run.sh NET` with NET = P or 1 to 5, one pod each.
Outputs: `protocols/14-heading-map/<NET>/`.

## Outcomes (no pass mark)

Per network and cue: localization (alive) per seed; the decoded bump angle in every 2-s
window relative to the cue angle; the absolute error in the last window (17.6 to 19.6 s);
the number of EPG neurons active. Summary per network: the number of the 24 trials with
localization above 0.5, and of those, the number whose last-window angle lies within 30
degrees of the cue.

## What would count as informative

If bumps persist at every cue and stay within 30 degrees of it, the networks hold a heading
at any position and the text can say so. If bumps persist only near the training sector, or
persist but move to one preferred position, the learned state is a single stable bump (a
point attractor on the ring), and the text must say that instead of "heading".
