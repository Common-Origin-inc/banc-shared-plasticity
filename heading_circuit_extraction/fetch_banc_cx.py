"""Fetch the central-complex heading circuit directly from BANC via CAVE.

No intermediary files, no derived matrices. Everything here is pulled live from
the public datastack and every selection rule is stated in the metadata rather
than inherited.

    datastack        brain_and_nerve_cord_public
    materialization  888 (latest)
    cell types       codex_annotations, classification_system = 'cell_type'
    synapses         synapses_v3
    transmitters     synapses_v2_nt_prediction_5

THE BOUNDARY PROBLEM
────────────────────
The six "core" heading types are a mechanism, not a closed system: EPG neurons
receive about two thirds of their synapses from neurons outside that set, and a
151-neuron extraction fires ~0.1 Hz because most of its drive was deleted.

There is no clean cut. Every boundary leaves neurons dangling, and following the
input backwards terminates only at photoreceptors. So this script draws the cut
EXPLICITLY: rank all upstream partners by synapses made onto the core, and take
from the top until INPUT_COVERAGE of the incoming synaptic mass is accounted
for. The threshold is a stated number, not a naming convention.

Both circuits are written to one file with an `is_core` mask, so the simulation
can run the 151-neuron and widened versions from the same data and we can find
out whether the extra neurons earn their cost.

SIGN POLICY (ours, recorded, not inherited)
    GABA, glutamate            -> inhibitory  (-1)
    everything else, incl. unknown -> excitatory (+1)
Shiu et al.'s convention, adopted because w_syn = 0.275 mV was calibrated under
it. For this circuit it is uncontroversial: the compass types are acetylcholine
and Delta7 is glutamate, so every alternative convention agrees.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent

DATASTACK = "brain_and_nerve_cord_public"
CORE_TYPES = ["EPG", "EPGt", "PEN_a(PEN1)", "PEN_b(PEN2)", "PEG", "Delta7"]
INPUT_COVERAGE = 0.85          # include upstream sources up to this share of input
MAX_UPSTREAM = 700             # hard cap, keeps the network tractable
INHIBITORY_NT = {"gaba", "glutamate"}


def log(msg):
    print(msg, flush=True)


def main() -> int:
    from caveclient import CAVEclient

    client = CAVEclient(DATASTACK)
    mat = client.materialize.most_recent_version()
    log(f"datastack {DATASTACK}   materialization {mat}")

    # ---------------------------------------------------------------- core
    ann = client.materialize.query_table(
        "codex_annotations",
        filter_in_dict={"classification_system": ["cell_type"],
                        "cell_type": CORE_TYPES})
    ann = ann.drop_duplicates("pt_root_id")
    core_ids = ann.pt_root_id.to_numpy()
    core_type = dict(zip(ann.pt_root_id, ann.cell_type.astype(str)))
    log(f"core: {len(core_ids)} neurons")
    for t in CORE_TYPES:
        log(f"  {t:<14} {sum(v == t for v in core_type.values()):>3}")

    # ------------------------------------------------- rank upstream inputs
    log("\nquerying synapses onto core ...")
    t0 = time.time()
    syn_in = client.materialize.query_table(
        "synapses_v3", filter_in_dict={"post_pt_root_id": core_ids.tolist()})
    log(f"  {len(syn_in):,} synapses   [{time.time()-t0:.1f}s]")

    ext = syn_in[~syn_in.pre_pt_root_id.isin(set(core_ids.tolist()))]
    rank = ext.groupby("pre_pt_root_id").size().sort_values(ascending=False)
    total_in = len(syn_in)
    frac_external = len(ext) / total_in
    log(f"  from outside the core: {len(ext):,} ({100*frac_external:.1f}%)"
        f"   from {rank.size:,} distinct upstream neurons")

    cum = rank.cumsum() / len(ext)
    n_take = int(np.searchsorted(cum.to_numpy(), INPUT_COVERAGE) + 1)
    n_take = min(n_take, MAX_UPSTREAM)
    upstream_ids = rank.index.to_numpy()[:n_take]
    achieved = float(cum.iloc[n_take - 1])
    log(f"  taking top {n_take} upstream neurons -> "
        f"{100*achieved:.1f}% of external input covered")

    union_ids = np.concatenate([core_ids, upstream_ids])
    log(f"\nunion circuit: {len(union_ids)} neurons")

    # ------------------------------------------------ synapses within union
    log("querying synapses within the union ...")
    t0 = time.time()
    syn = client.materialize.query_table(
        "synapses_v3",
        filter_in_dict={"pre_pt_root_id": union_ids.tolist(),
                        "post_pt_root_id": union_ids.tolist()})
    log(f"  {len(syn):,} synapses   [{time.time()-t0:.1f}s]")

    counts = (syn.groupby(["pre_pt_root_id", "post_pt_root_id"])
                 .size().rename("n").reset_index())
    log(f"  {len(counts):,} connected pairs   "
        f"mean {counts.n.mean():.2f} synapses/pair")

    # ------------------------------------------------------- transmitters
    # The base table `synapses_v2_nt_prediction_5` rejects a pre_pt_root_id
    # filter with a 500 (it is a reference table on synapses_v2). Its VIEW does
    # accept the filter, and exposes `type` = transmitter, `value` = confidence.
    # Chunked because one query for 851 neurons would return millions of rows;
    # a few hundred synapses per neuron is far more than enough to settle the
    # modal transmitter.
    log("\nfetching neurotransmitter predictions (chunked view query) ...")
    nt_by_neuron: dict[int, str] = {}
    CHUNK, PER_CHUNK_LIMIT = 40, 8000
    t0 = time.time()
    for k in range(0, len(union_ids), CHUNK):
        batch = union_ids[k:k + CHUNK].tolist()
        try:
            v = client.materialize.query_view(
                "synapses_v2_nt_prediction_5_human_readable",
                filter_in_dict={"pre_pt_root_id": batch},
                limit=PER_CHUNK_LIMIT)
            if len(v):
                modal = (v.groupby("pre_pt_root_id")["type"]
                          .agg(lambda s: s.value_counts().idxmax()))
                nt_by_neuron.update({int(a): str(b).lower()
                                     for a, b in modal.items()})
        except Exception as exc:
            log(f"  chunk {k // CHUNK} failed: {type(exc).__name__}")
        if (k // CHUNK) % 5 == 0:
            log(f"  {k + len(batch):>4}/{len(union_ids)} neurons   "
                f"{len(nt_by_neuron)} typed   [{time.time()-t0:.0f}s]")
    log(f"  transmitters resolved for {len(nt_by_neuron)}/{len(union_ids)} "
        f"neurons   [{time.time()-t0:.0f}s]")

    # ------------------------------------------------ volumes and areas
    # Membrane surface area sets the PSP size: dV = Q / (c_m * A_i). Volume is
    # kept for the fragment test (< 10 um^3 = not a whole neuron).
    #
    # BANC exposes neither as a column: cell_info has none, the somas tables
    # carry only positions, and l2cache.get_l2data() on a ROOT id returns {}.
    # Both have to be assembled from the segmentation itself -- fetch each
    # neuron's level-2 chunks, then sum their size_nm3 / area_nm2. That is
    # what the l2 cache is for.
    log("\nfetching neuron volumes and membrane areas from l2cache ...")
    t0 = time.time()
    l2_by_neuron: dict[int, np.ndarray] = {}
    for k, r in enumerate(union_ids):
        try:
            l2_by_neuron[int(r)] = client.chunkedgraph.get_leaves(int(r),
                                                                  stop_layer=2)
        except Exception:
            l2_by_neuron[int(r)] = np.array([], dtype=np.int64)
        if k % 100 == 0:
            log(f"  leaves {k:>4}/{len(union_ids)}   [{time.time()-t0:.0f}s]")

    all_l2 = np.unique(np.concatenate([v for v in l2_by_neuron.values()
                                       if len(v)]))
    log(f"  {len(all_l2):,} distinct L2 chunks; querying sizes ...")
    size_lut: dict[int, float] = {}
    area_lut: dict[int, float] = {}
    BATCH = 5000
    for k in range(0, len(all_l2), BATCH):
        chunk = all_l2[k:k + BATCH].tolist()
        try:
            d = client.l2cache.get_l2data(chunk,
                                          attributes=["size_nm3", "area_nm2"])
            size_lut.update({int(a): float(b.get("size_nm3", 0.0))
                             for a, b in d.items() if b})
            area_lut.update({int(a): float(b.get("area_nm2", 0.0))
                             for a, b in d.items() if b})
        except Exception as exc:
            log(f"  size batch {k // BATCH} failed: {type(exc).__name__}")
        if (k // BATCH) % 20 == 0:
            log(f"  sizes {k:>7}/{len(all_l2)}   [{time.time()-t0:.0f}s]")

    def per_neuron(lut):
        return np.array([sum(lut.get(int(x), 0.0) for x in l2_by_neuron[int(r)])
                         for r in union_ids], dtype=np.float64)

    volume_nm3 = per_neuron(size_lut)
    area_nm2 = per_neuron(area_lut)
    n_missing = int((volume_nm3 <= 0).sum())
    log(f"  volumes for {len(volume_nm3) - n_missing}/{len(union_ids)} neurons"
        f"   [{time.time()-t0:.0f}s]")
    if n_missing:
        med = float(np.median(volume_nm3[volume_nm3 > 0]))
        volume_nm3[volume_nm3 <= 0] = med
        log(f"  {n_missing} neurons had no volume; set to median {med:.3e} nm3")

    vol_um3 = volume_nm3 / 1e9
    log(f"  volume um^3: median {np.median(vol_um3):.1f}   "
        f"min {vol_um3.min():.1f}   max {vol_um3.max():.1f}   "
        f"ratio max/min {vol_um3.max()/max(vol_um3.min(),1e-9):.1f}x")

    # Fragments: a "neuron" under 10 um^3 is a piece of segmentation, not a
    # whole cell. Its measured area is far below the cell's true area and
    # 1/A would give it an absurd PSP. Assign the median area of intact
    # neurons and flag it, so the assumption is visible downstream.
    FRAGMENT_UM3 = 10.0
    is_fragment = (vol_um3 < FRAGMENT_UM3) | (area_nm2 <= 0)
    med_area = float(np.median(area_nm2[~is_fragment]))
    area_nm2[is_fragment] = med_area
    area_um2 = area_nm2 / 1e6
    log(f"  area um^2 (intact): median {np.median(area_um2[~is_fragment]):.0f}   "
        f"min {area_um2[~is_fragment].min():.0f}   "
        f"max {area_um2[~is_fragment].max():.0f}")
    log(f"  {int(is_fragment.sum())} fragments (< {FRAGMENT_UM3:g} um^3) set "
        f"to median area {med_area/1e6:.0f} um^2")

    # ------------------------------------------------------- build matrix
    idx = {int(r): i for i, r in enumerate(union_ids)}
    n = len(union_ids)
    W = np.zeros((n, n), dtype=np.float32)

    nt_arr, sign_arr = [], []
    for r in union_ids:
        nt_name = nt_by_neuron.get(int(r), "unknown")
        nt_arr.append(nt_name)
        sign_arr.append(-1.0 if nt_name in INHIBITORY_NT else 1.0)
    sign_arr = np.array(sign_arr, dtype=np.float32)
    nt_arr = np.array(nt_arr)

    for pre, post, cnt in counts.itertuples(index=False):
        i, j = idx.get(int(pre)), idx.get(int(post))
        if i is not None and j is not None:
            W[i, j] = cnt * sign_arr[i]          # signed by PRESYNAPTIC neuron

    is_core = np.array([int(r) in core_type for r in union_ids])
    labels = np.array([core_type.get(int(r), "upstream") for r in union_ids])

    log(f"\nmatrix {W.shape}   {int((W != 0).sum()):,} edges   "
        f"{int(np.abs(W).sum()):,} synapses")
    log(f"  core neurons {is_core.sum()}   upstream {(~is_core).sum()}")
    from collections import Counter
    log(f"  transmitters: {dict(Counter(nt_arr).most_common(6))}")
    log(f"  excitatory {int((sign_arr > 0).sum())}   "
        f"inhibitory {int((sign_arr < 0).sum())}")

    # ------------------------------ input coverage inside the widened circuit
    core_local = np.flatnonzero(is_core)
    in_union = np.abs(W[:, core_local]).sum()
    log(f"\ncore input now captured inside the circuit: "
        f"{in_union:,.0f} of {total_in:,} synapses "
        f"({100*in_union/total_in:.1f}%)   was "
        f"{100*(1-frac_external):.1f}% with core only")

    # ----------------------------------------------- ring geometry (analysis)
    # Features are taken over the CORE submatrix only. Including the 700
    # upstream neurons dilutes the EPG-specific structure and degrades the
    # embedding (radius cv 0.10 -> 0.31, antipodal similarity 0.05 -> 0.40).
    epg_full = np.flatnonzero(labels == "EPG")          # index into all 851
    core_cols = np.flatnonzero(is_core)
    A_full = np.abs(W)
    A = A_full[np.ix_(core_cols, core_cols)]            # core-only submatrix
    epg = np.array([int(np.flatnonzero(core_cols == e)[0])
                    for e in epg_full])                 # index into the core
    feat = np.hstack([A[epg, :], A[:, epg].T])
    feat = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-12)
    sim = feat @ feat.T
    np.fill_diagonal(sim, 0.0)
    deg = sim.sum(1)
    dinv = np.diag(1 / np.sqrt(deg + 1e-12))
    _, vecs = np.linalg.eigh(dinv @ (np.diag(deg) - sim) @ dinv)
    ang_epg = np.arctan2(vecs[:, 2], vecs[:, 1])
    radius = np.hypot(vecs[:, 1], vecs[:, 2])
    cv = float(radius.std() / radius.mean())

    srt = np.argsort(ang_epg)
    m = len(epg)
    prof = [float(np.mean([sim[srt[i], srt[(i + d) % m]] for i in range(m)]))
            for d in range(1, m // 2 + 1)]
    log(f"\nring recovery ({m} EPG): radius cv {cv:.3f}   "
        f"sep1 {prof[0]:.3f} -> antipodal {prof[-1]:.3f}")

    # propagate back in FULL space: every other neuron takes the synapse-weighted
    # circular mean of the EPG angles it wires with
    angles = np.full(n, np.nan)
    angles[epg_full] = ang_epg
    unit = np.exp(1j * ang_epg)
    for i in range(n):
        if labels[i] == "EPG":
            continue
        w = A_full[i, epg_full] + A_full[epg_full, i]
        if w.sum() > 0:
            angles[i] = np.angle((w * unit).sum())

    # ------------------------------------------------------------- save
    np.savez_compressed(
        OUT / "banc_cx_circuit.npz",
        W=W, is_core=is_core, labels=labels, angles=angles,
        sign=sign_arr, nt=nt_arr, root_id=union_ids.astype(str),
        volume_nm3=volume_nm3, area_nm2=area_nm2, is_fragment=is_fragment,
    )
    meta = {
        "datastack": DATASTACK,
        "materialization": int(mat),
        "fetched_utc": pd.Timestamp.utcnow().isoformat(),
        "core_types": CORE_TYPES,
        "core_selection": "codex_annotations, classification_system='cell_type'",
        "synapse_table": "synapses_v3",
        "n_core": int(is_core.sum()),
        "n_upstream": int((~is_core).sum()),
        "n_total": n,
        "n_edges": int((W != 0).sum()),
        "n_synapses": float(np.abs(W).sum()),
        "boundary": {
            "rule": "rank upstream neurons by synapses onto core, take top-N "
                    "until INPUT_COVERAGE of external input mass",
            "input_coverage_target": INPUT_COVERAGE,
            "input_coverage_achieved": achieved,
            "max_upstream_cap": MAX_UPSTREAM,
            "n_upstream_taken": int(n_take),
            "core_input_from_outside_core_pct": 100 * frac_external,
            "core_input_captured_in_circuit_pct": float(100 * in_union / total_in),
        },
        "sign_policy": {
            "inhibitory_transmitters": sorted(INHIBITORY_NT),
            "default": "excitatory",
            "note": "Shiu et al. convention; w_syn = 0.275 mV was calibrated "
                    "under it",
        },
        "ring_recovery": {
            "radius_cv": cv, "similarity_vs_separation": prof,
            "note": "ANALYSIS METADATA ONLY, never passed to the simulation",
        },
        "volumes": {
            "source": "l2cache size_nm3 summed over each neuron's level-2 "
                      "chunks (chunkedgraph.get_leaves stop_layer=2)",
            "n_missing_set_to_median": n_missing,
            "median_um3": float(np.median(vol_um3)),
            "min_um3": float(vol_um3.min()),
            "max_um3": float(vol_um3.max()),
            "by_type_median_um3": {
                t: float(np.median(vol_um3[labels == t]))
                for t in CORE_TYPES + ["upstream"]
                if (labels == t).any()
            },
        },
        "areas": {
            "source": "l2cache area_nm2 summed over each neuron's level-2 "
                      "chunks; sets PSP size via dV = Q / (c_m * A)",
            "fragment_rule": f"volume < {FRAGMENT_UM3:g} um^3 => area set to "
                             "median of intact neurons, is_fragment=True",
            "n_fragments": int(is_fragment.sum()),
            "median_intact_um2": float(np.median(area_um2[~is_fragment])),
            "min_intact_um2": float(area_um2[~is_fragment].min()),
            "max_intact_um2": float(area_um2[~is_fragment].max()),
            "by_type_median_um2": {
                t: float(np.median(area_um2[(labels == t) & ~is_fragment]))
                for t in CORE_TYPES + ["upstream"]
                if ((labels == t) & ~is_fragment).any()
            },
        },
    }
    with open(OUT / "banc_cx_circuit.json", "w") as f:
        json.dump(meta, f, indent=2)

    log(f"\nwrote {OUT / 'banc_cx_circuit.npz'}")
    log(f"wrote {OUT / 'banc_cx_circuit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
