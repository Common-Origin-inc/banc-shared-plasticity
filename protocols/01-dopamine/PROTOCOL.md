# Protocol 01: modulation-source identity

Written before the audit is run.

## Question

`tools/build_da_edges.py` marks a neuron as a modulation source when the string
`dopamine` appears anywhere in `neurotransmitter_verified`. `tools/build_banc_full.py`
assigns the fast transmitter from the first comma-separated token of the same column.
Ten stored sources are therefore both a fast source and a modulation source. Four of
them are ER4m neurons that supply 2,355 of the 4,573 modulation contacts onto EPG (51.5%).

Do the raw annotations support treating these ten neurons as dopamine sources?

## Input

- The BANC metadata table `banc_888_meta.feather` (see the README).
  Expected SHA-256:
  `86ccf5df0c67419f8c5f43e93a7ed38d23a080e9f7fde26737290252f3780098`, 57,503,026 bytes.
- `BANC_BEST_BRAIN/da_edges.npz` and `BANC_BEST_BRAIN/banc_full.npz` as committed.
- The `root_888` identifiers of the ten dual-classified sources (listed in `audit.py`).

## Steps

1. Download the feather file and verify its SHA-256 and size against the values above. Stop if they differ.
2. For the ten identifiers, record verbatim: `neurotransmitter_verified`, the predicted
   transmitter column(s), cell type, super class, and every column that states where an
   annotation came from.
3. For all 380 modulation sources in `da_edges.npz`, tabulate the full
   `neurotransmitter_verified` string and count: dopamine as the only token, dopamine as
   the first of several, dopamine not first.
4. Recompute from the committed arrays, without trusting the manuscript: modulation
   contacts onto the 45 ring EPG neurons in total, and the share from each of the ten
   dual-classified sources.
5. Recompute what the modulation table would contain under two alternative selectors:
   (a) first token equals `dopamine`; (b) `dopamine` is the only token. Report sources,
   edges, contacts, and contacts onto EPG for each.

## Outputs

- `protocols/01-dopamine/audit.json`: file hash, the verbatim rows, the counts.
- `protocols/01-dopamine/AUDIT.md`: the same in readable form, with the script that produced it.
- `protocols/01-dopamine/audit.py`: the script. Read-only on all inputs.

No model file, graph, checkpoint or result is modified by this protocol.

## Decision rule

The audit reports what the annotations say. It does not decide the biology. The decision
was made before protocol 02 and is stated there:

- **Keep** the current checkpoint if the annotations give an explicit dopamine label with
  a stated source for all four ER4m neurons. The paper keeps the wording "anatomically
  assigned modulation" and reports the dual use.
- **Rebuild** otherwise: regenerate the modulation table under selector (a) into a new
  file, retrain stage 2 from the existing stage-1 checkpoint into a new checkpoint, and
  keep `w_final_p37a.npz` as the historical artifact. Replication runs (protocol 02) then
  use the rebuilt table.

Protocols 02 onward do not start until this decision is recorded.
