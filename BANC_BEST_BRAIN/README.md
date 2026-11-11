# BANC_BEST_BRAIN

A whole-brain spiking model of *Drosophila* built from the BANC connectome (brain + ventral nerve
cord, 76,683 neurons, 27.7 M synapses) under one rule: **no parameter is set per cell type**.
Weights come from synapse counts, membrane properties from measured surface areas, and every
constant applies to every neuron or synapse alike. The brain is developed for 600 s of simulated
time by two universal plasticity rules, frozen, and then judged on 24 held-out circuit experiments
spanning the heading system, feeding, escape, olfaction and the resting state.

This folder holds the **final best brain** and everything needed to reproduce and re-test it:
the frozen weights, the model, the settle scripts, the full test suite, and the trial results.
The development history that led here (40+ settles, the negative results, the ideas that failed)
is not in this repository.

## The best brain in one paragraph

Static count-weighted model (v2) → 100 s of inhibitory plasticity (Vogels) + excitatory rate cap
→ 500 s of inhibitory plasticity + dopamine-gated STDP, where the dopamine is the connectome's own
dopaminergic synapses → frozen. Run with 15 ms excitatory synapses, a 1 mV soft threshold, and
short-term depression on excitatory→excitatory synapses whose release fraction is scaled by the
edge's reciprocity in the connectome (recurrent loops depress, feed-forward chains do not).
Details: [METHOD.md](METHOD.md). Every component was ablated and is necessary
([RESULTS.md](RESULTS.md#ablation)).

## Headline results

| | static v2 | best brain |
|---|---|---|
| resting state (non-sensory neurons) | 25–31k active @ 31–44 Hz, 5–7 % at the refractory ceiling (seizure) | 13–17k @ 4.8–6.7 Hz, ISI CV 1.0, 0.2 % at ceiling |
| heading bump (Kim et al. 2017 protocol) | 270–290 Hz, rigid, clock-like | **22–33 Hz, persists 20 s on 5/5 seeds**, 56–80° wide, drive-invariant, forms without a cue |
| odour coding (Kenyon cells) | 45–80 % active, mushroom body ignites | 3–9 % active, two odours nearly disjoint (Jaccard 0.01), graded with concentration |
| looming → giant fibre | 44–52 Hz, specific | 16 Hz, specific |
| sugar → proboscis motor neuron | 150–182 Hz | 0 (lost in the homeostatic settle) |
| **24-test scorecard** | **5 / 24** | **17 / 24** |

Both brains fail rotation of the bump under PEN drive, capture by a second cue, competition
between two cues, bitter suppression of sugar, and water → motor neuron. The best brain's other
open problems: bump narrower than the fly's 90°, sugar reflex lost, and under the uniform
resting-state background the mushroom body is more active than a resting fly's. Full tables and
criteria: [RESULTS.md](RESULTS.md); what each test does: [TESTS.md](TESTS.md).

## Quick start

```bash
pip install -r requirements.txt            # brian2, numpy, pandas, matplotlib (+ brian2cuda for a GPU)

# the best brain's full scorecard (GPU ~15 min; CPU cpp_standalone hours; add --device runtime for pure Python)
python validate_scorecard.py --gains w_final_p37a.npz --escape 1 --tau-e 15 --vis --std 0.2,0.2,recip_ee \
    --only kim,kimrest,pen,feeding,looming,loom1,olfaction,olf4,rest --seed 1 --device cuda_standalone

# one capability test, e.g. 20 s persistence of the bump in the resting brain
python capability_tests.py --brain p37a --std 0.2,0.2,recip_ee --test persist --seed 1 --seconds 20 --device cuda_standalone

# the static v2 control on the same test
python capability_tests.py --brain v2 --test persist --seed 1 --seconds 20 --device cuda_standalone

# everything (scorecard, capability battery, LFT, ablations) -- see reproduce.sh and run_suite.sh
bash reproduce.sh cuda_standalone score
```

Each standalone run compiles one program (~40–90 s on an RTX 4090) and prints one result line
(`KIM ...`, `CAP ...`, `LFT ...`) plus a JSON. `--device runtime` needs no compiler but is ~100×
slower; a 4 s test then takes ~10 min on a laptop.

## Reproducing the brain from the connectome

```bash
bash reproduce.sh cuda_standalone p8      # 100 s homeostatic settle from the v2 weights  -> w_final_p8.npz   (~5 min GPU)
bash reproduce.sh cuda_standalone p37a    # 500 s settle with dopamine-gated STDP from p8  -> w_final_p37a.npz (~25 min GPU)
bash reproduce.sh cuda_standalone score   # scorecard on the frozen brain with depression on
bash reproduce.sh cuda_standalone suite   # capability battery + LFT + ablations (~3 h GPU)
```

Both weight files are committed, so `score` and `suite` alone reproduce every table in RESULTS.md.
The settle draws a random seed (p37a used 11); the mushroom body ignites on some draws (a known,
documented coin flip), so a re-learned brain may differ.

Building `banc_full.npz` and `da_edges.npz` from the raw BANC tables needs the Lee-lab bulk
download (`banc_888_meta.feather`, `banc_888_edgelist_simple_v3.feather`, the neurotransmitter
prediction table, L2 morphology parquet files) in a folder named by `BANC_BULK`; see `tools/`.
Both derived files are committed, so this step is optional.

## Files

| file | what |
|---|---|
| `plasticity.py` | the model: neuron/synapse equations, the v2 weight rule, both plasticity rules, dopamine gating, escape noise, short-term depression, `edge_reciprocity` |
| `banc_common.py` | neuron constants, Kim protocol timing, ring-bump decoding (population vector, R, width) |
| `run_native_plasticity.py` | the settle runner (stages p8 and p37a; every flag is recorded in the saved weights' `params`) |
| `validate_scorecard.py` | the ten original experiments (Kim silent/resting, PEN rotation, feeding ×4, looming + single LC types, olfaction ×4 glomeruli, rest); `--std`, `--live`, `--gains-class`, `--pen-seed` |
| `capability_tests.py` | persistence (20 s), second cue, competition, drive invariance, resting statistics, spontaneous bump, two-odour separability, concentration series, evoked-activity decay; `--precheck` prints the depression rule's U by pathway |
| `lft_bump.py` | maximum-entropy (Ising) coupling fingerprint of the ring raster |
| `ablate_check.py` | pass/fail of one ablation (used by `run_suite.sh ablate`) |
| `gains_by_type.py` | learned gain by cell-type pair (loop E, ExR6/Δ7/ER inhibition, KC→KC, PN→KC) |
| `run_suite.sh` | batch runner for every experiment family (`full`, `ablate`, `suite2`, `pen`, `recip`, ...) |
| `reproduce.sh` | the pipeline end to end |
| `tools/` | graph builders from the BANC bulk tables; RunPod setup script |
| `banc_full.npz` | the graph: 76,683 neurons (root id, cell type, super class, side, transmitter sign, area), 4.8 M excitatory + inhibitory edges with synapse counts, CX core mask |
| `banc_cx_circuit.npz` | the 45 ring EPGs' order and angles (for decoding the bump) |
| `da_edges.npz` | the 33,054 dopaminergic edges (380 verified DA neurons) that gate E→E plasticity |
| `w_final_p8.npz`, `w_final_p37a.npz` | learned per-edge gains after stage 1 and stage 2 (+ the settle parameters as JSON) |
| `results/scorecards/` | scorecard JSONs: best brain (full + kimrest seeds), uniform/all-E depression variants, the seven ablations, the gated-brake lineage (p8r, p37ar) |
| `results/capability/` | 178 capability JSONs (best brain, v2, ablations, variants) and the complete run log `cap.log` |
| `results/lft/` | LFT fingerprints (npz/json/png) for v2, p37a, the hand fit and the best brain |

New runs write next to the code — `scorecard_<tag>.json`, `results_cap/`, `results_lft/`,
`cap.log` — and leave the archived trial results under `results/` untouched.

## Model summary

Conductance-based leaky integrate-and-fire neurons: C = 1 µF/cm² × surface area, τ_m 20 ms,
rest/reset −52 mV, threshold −45 mV (soft: hazard (1/τ_m)·exp((V−V_th)/1 mV)), refractory
2.2 ms, E_exc 0 mV, E_inh −70 mV, τ_inh 5 ms, τ_exc 15 ms (charge-preserved). Every excitatory
synapse: 30 pS × synapse count × learned gain; every inhibitory synapse: scaled per neuron so that
inhibition balances excitation at threshold, × learned gain. Transmitter sign from the BANC
annotations (GABA/glutamate/histamine inhibitory, acetylcholine excitatory; modulator neurons'
fast synapses dropped). Excitatory→excitatory synapses carry Tsodyks–Markram depression with
U_ij = 0.2 · n_ji/(n_ij + n_ji), τ_rec 0.2 s. Spontaneous drive: sensory and visual-projection
neurons fire at 10 Hz. Eleven universal constants were swept in this project (listed in
METHOD.md); none is per cell type.

## Citation of sources

BANC connectome: Lee lab compiled tables (banc_888). Model lineage: Shiu et al. 2024 (count-weighted
LIF); Kakaria & de Bivort 2017 (neuron constants); Vogels et al. 2011 (inhibitory plasticity);
Izhikevich 2007 / Frémaux & Gerstner 2016 (three-factor STDP); Tsodyks & Markram 1997 (depression);
Kim et al. 2017 (bump protocol); Franchini & Bardella 2026 (LFT fingerprint).
