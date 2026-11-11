# From wiring to function: shared local plasticity turns a fly connectome into a functional spiking network

Code, data, protocols and results for the preprint by Srijith Chinthalapudi, Filippo Piazza and
Gioele Nanni (Common Origin). The manuscript is in [`paper/`](paper/), with its PDF, figure script, a
map from every number to the file it comes from ([`paper/NUMBERS.md`](paper/NUMBERS.md)).

**Model.** A conductance-based leaky integrate-and-fire model of the BANC brain-and-nerve-cord
connectome of *Drosophila* (76,683 neurons, 27.7 M synaptic contacts):
- membrane areas come from electron-microscopy meshes, and no parameter is set per cell type;
- synaptic gains are set by two learning stages whose rules are shared by every synapse of a class;
- evaluation adds short-term depression scaled by the reciprocity of each connection.

**What it shows.**
- In five of five independent runs the learned network rests at 6.5 to 9.4 Hz (43 Hz with static
  weights), holds a single localized bump of compass-neuron (EPG) activity at moderate rates, and
  transmits looming input to the giant fiber.
- The learned bump is stable but not yet movable: it sits at one preferred position per network
  and returns there after a cue at any other position.
- In four of five runs the bump depends on modulatory gating assigned to four ER4m neurons, much
  more than on a matched share of other modulatory input (LPsP).

## Layout

| path | content | used for |
|---|---|---|
| `paper/` | manuscript in the IEEE conference format for NICE 2027 (`manuscript.tex`, `manuscript.pdf`, `references.bib`), the earlier 11-pt single-column version before the pre-submission corrections (`longer_manuscript.tex`, `longer_manuscript.pdf`), `make_figures.py`, `figures/` (with `evidence.json`: SHA-256 of every file read), `NUMBERS.md` | everything |
| `BANC_BEST_BRAIN/` | the whole-network model: graph (`banc_full.npz`), modulation tables (`da_edges*.npz`), learned weights of network P (`w_final_p8.npz`, `w_final_p37a.npz`), learning (`run_native_plasticity.py`, `plasticity.py`), tests (`capability_tests.py`, `validate_scorecard.py`), `reproduce.sh`, archived results (`results/`) | Results 2.2 to 2.7 |
| `heading_circuit_extraction/` | the 851-neuron heading-circuit extraction and its synapse-by-leak ablation | Results 2.1, Fig. 1 |
| `protocols/01-dopamine` | audit of the modulation sources; rebuilt tables (ER4m-, LPsP-removed, first label) | 2.6 |
| `protocols/02-replication` | five independent learning runs, fixed commands (`run.sh`), all results and run log | 2.2, Table 1, Figs. 2 and 3 |
| `protocols/03-shiu-baseline` | the current-based formulation of Shiu et al. on the same graph | 2.7 |
| `protocols/04-static-rerun` | static-weight references H and M | Table 1, Fig. 2 |
| `protocols/07-rotating-kicks` | training with kicks rotating through eight sectors | Discussion |
| `protocols/13-er4m-only` | retraining without the four ER4m sources | 2.6, Fig. 3 |
| `protocols/14-heading-map` | persistence after a cue at eight ring positions; activity traces | 2.2 |
| `protocols/15-lpsp-only` | retraining without the two LPsP sources (control) | 2.6, Fig. 3 |
| `protocols/16-static-bump` | the bump in the static brain-and-cord network after a localized cue | 2.1 |

Each protocol folder holds `PROTOCOL.md` (written before its runs), the commands, the result files and
`RESULTS.md`.
Protocols are numbered in the order they were run; those the paper does not use are not included.

## Learned weights

Network P's checkpoints are in `BANC_BEST_BRAIN/`. The checkpoints of the five independent runs
(stage 1 `w_final_rep<k>_p8.npz` and stage 2 `w_final_rep<k>_p37a.npz`) and of the LPsP-removed runs
(`w_final_rep<k>l_p37a.npz`) are attached to the GitHub release `weights-v1` (about 170 MB) with
`SHA256SUMS`. Place them in `BANC_BEST_BRAIN/` to use them with the commands below. They were
produced by protocol 14 and 15 runs; retraining from scratch reproduces the protocol 02 outcomes
exactly on the same GPU model.

## Reproducing

Requirements:
- an NVIDIA GPU (all results: RTX 4090);
- Python 3.12, brian2 2.10.1, brian2cuda 1.0b1, numpy 2.5.3, pandas, matplotlib
  (`BANC_BEST_BRAIN/tools/setup_runpod.sh` sets this up on a fresh Ubuntu GPU machine);
- for the feeding test of the scorecard only (not used by the paper): the BANC metadata table
  `banc_888_meta.feather` from the BANC data release, in `bulk/` at the repository root (or set
  `BANC_BULK` to its folder). Without it the scorecard skips feeding and runs every other test. The
  graph-building scripts in `BANC_BEST_BRAIN/tools/` read the other BANC tables from the same place.

```bash
# one independent run k (stage 1, stage 2, full evaluation), about 70 min on an RTX 4090
bash protocols/02-replication/run.sh sub 1

# the same with the ER4m or the LPsP sources removed from the modulation table
bash protocols/02-replication/run.sh noer4m 1
bash protocols/02-replication/run.sh nolpsp 1

# persistence after a cue at eight positions for network P or run k (retrains run k if its checkpoint is absent)
bash protocols/14-heading-map/run.sh P

# bump in the static network (Results 2.1), about 10 min
bash protocols/16-static-bump/run.sh

# network P from the included graph: stage 1, stage 2, scorecard (the "data" stage rebuilds the graph
# from the BANC tables in bulk/, which are not redistributed here)
bash BANC_BEST_BRAIN/reproduce.sh cuda_standalone p8
bash BANC_BEST_BRAIN/reproduce.sh cuda_standalone p37a
bash BANC_BEST_BRAIN/reproduce.sh cuda_standalone score

# figures (reads archived result files only; matplotlib >= 3.7)
python paper/make_figures.py

# unit tests (CPU)
python -m pytest BANC_BEST_BRAIN/tests
```

## Licence and data

- **Code, protocols and text:** MIT ([`LICENSE`](LICENSE)).
- **Files derived from BANC** (graphs, modulation tables and the learned weights indexed by them):
  CC BY 4.0, the licence of the BANC data release ([`DATA_NOTICE.md`](DATA_NOTICE.md)).

If you use this work, cite the preprint ([`CITATION.cff`](CITATION.cff)) and the BANC connectome:
Bates et al., *Nature* 656, 957–970 (2026).
