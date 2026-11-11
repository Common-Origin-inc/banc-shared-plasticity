# Protocol 07: training kicks that rotate around the ring

Written before any run (2026-10-01).

## Question

In every learned network so far the training stimulus kicks the same 45 degree EPG sector
(stage 1 every 10 s, stage 2 every 3 s). All of them hold a bump and none rotates it smoothly
under PEN drive, moves it to a second cue, or resolves two cues reliably. Does cycling the
kick through sectors around the ring change those outcomes, and what does it cost elsewhere?

## Change

One flag that already exists in `run_native_plasticity.py`: `--bump-rotate 8`, in both stages.
Kicks cycle through 8 sectors spaced 45 degrees apart. Kick timing, duration, strength and
number are unchanged, so total stimulation is the same as in protocol 02. Nothing else
changes: same code, same modulation table (`da_edges.npz`), same seeds as protocol 02
(stage 1: 101 to 105, stage 2: 201 to 205), same evaluation commands and evaluation seeds.
The evaluation protocols still kick the standard sector.

## Commands

`ROT=8 bash protocols/02-replication/run.sh sub K` for K = 1 to 5, one pod each.
Outputs: `protocols/02-replication/out_sub_rot8/` on each pod, collected to
`protocols/07-rotating-kicks/k<K>/`.

## Outcomes

Reported as measured values per replicate, next to the protocol 02 replicate with the same
seeds. No pass mark.

- Angular updating: PEN left and right shift and trajectory; second-cue shift at 90 and 180
  degrees for both cue strengths; number of bumps after two opposed cues, both seeds.
- Persistence: the protocol 02 primary outcomes (window-active EPG rate, alive, width, bump rate).
- Elsewhere: resting rate and ceiling fraction, Kenyon-cell recruitment, giant-fibre rate,
  MN9 sugar rate.

## What would count as informative

Either direction is a result. If updating improves without losing persistence, the fixed
sector was the cause of the pinned bump. If it does not, the limitation is not the curriculum.
If persistence degrades, the bump in protocol 02 depended on repeatedly training one location.
The mushroom-body outcome is expected to stay variable and is not the target here.
