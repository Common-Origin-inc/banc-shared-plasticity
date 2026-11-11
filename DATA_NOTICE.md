# Data notice

The neuron, connection and annotation data used here come from the BANC (brain-and-nerve-cord)
connectome of *Drosophila melanogaster*, materialization 888:

> Bates, A. S., Phelps, J. S., Kim, M., Yang, H. H. et al. Distributed control circuits across a
> brain-and-cord connectome. *Nature* 656, 957–970 (2026). https://doi.org/10.1038/s41586-026-10735-w

The BANC data release is licensed under the Creative Commons Attribution 4.0 International License
(CC BY 4.0, https://creativecommons.org/licenses/by/4.0/), as stated for its archived datasets on the
Harvard Dataverse (https://doi.org/10.7910/DVN/7WTH1N; preprint version https://doi.org/10.7910/DVN/8TFGGB).

The following files in this repository are adapted from BANC data and are distributed under CC BY 4.0,
not under this repository's MIT License:

- `BANC_BEST_BRAIN/banc_full.npz`, `BANC_BEST_BRAIN/banc_cx_circuit.npz`,
  `BANC_BEST_BRAIN/da_edges*.npz`, `heading_circuit_extraction/banc_cx_circuit.npz`;
- the learned weights (`BANC_BEST_BRAIN/w_final_*.npz` and the checkpoints attached to the release
  `weights-v1`), which are indexed by the BANC graph.

**Changes made to the BANC data.** The files above are not the BANC tables themselves. They keep a
subset of neurons (central brain, visual projection and centrifugal neurons, and ventral nerve cord;
for the extraction, the heading-circuit core types and their main upstream partners), aggregate
synapses into neuron-to-neuron contact counts, assign a fast-transmitter sign to each neuron,
attach membrane areas measured from the level-2 mesh cache, and build tables of modulatory
contacts. The learned weights are gains computed on these graphs by the simulations in this
repository. The build scripts in `BANC_BEST_BRAIN/tools/` and `heading_circuit_extraction/` give
every step.

If you use these files, cite the BANC publication above and keep this notice. The raw BANC tables
(`banc_888_*.feather`) are not redistributed here; they are available from the BANC data release.
