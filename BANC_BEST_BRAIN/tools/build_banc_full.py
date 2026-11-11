"""Build the unified BANC Brain + VNC network (banc_full.npz) from the BANC
compiled tables in $BANC_BULK (default: bulk/ at the repository root; read-only):
transmitter rule, verified-modulator synapse drop, sign rule, L2-measured areas
with the volume x length proxy as fallback, cable attenuation, on the neuron set

    region ∈ {central_brain, ventral_nerve_cord}
      ∪ super_class ∈ {visual_projection, visual_centrifugal}
      − super_class ∈ {glia, not_a_neuron, trachea}

Measured areas: bulk/l2_full.parquet (central brain) plus bulk/l2_vnc.parquet
(VNC) when present; otherwise the proxy calibrated on the 505 CX neurons (K).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1]                        # BANC_BEST_BRAIN/
BULK = Path(os.environ.get("BANC_BULK", OUT.parent / "bulk"))     # BANC tables, default bulk/ at the repository root
CX_NPZ = OUT / "banc_cx_circuit.npz"


class BB:                                          # neuron-set and transmitter rules
    KEEP_SUPER = {"visual_projection", "visual_centrifugal"}
    DROP_SUPER = {"glia", "not_a_neuron", "trachea"}
    INHIBITORY_NT = {"gaba", "glutamate", "histamine"}
    FAST_NT = {"acetylcholine", "gaba", "glutamate", "histamine"}

KEEP_REGION = {"central_brain", "ventral_nerve_cord"}


def log(msg):
    print(msg, flush=True)


def main() -> int:
    meta = pd.read_feather(BULK / "banc_888_meta.feather")
    edges = pd.read_feather(BULK / "banc_888_edgelist_simple_v3.feather")
    meta["root_id"] = meta.root_888.astype(np.int64)
    log(f"meta {len(meta):,} neurons   edgelist {len(edges):,} pairs")

    keep = meta.region.isin(KEEP_REGION) | meta.super_class.isin(BB.KEEP_SUPER)
    keep &= ~meta.super_class.isin(BB.DROP_SUPER)
    n = meta[keep].copy().reset_index(drop=True)
    log(f"neuron set: {len(n):,}  (" + ", ".join(f"{k} {v:,}" for k, v in n.region.fillna('NA').value_counts().items()) + ")")

    # ---------------------------------------------------------- transmitter (as build_banc_brain)
    votes = pd.read_csv(BULK / "banc_888_neurotransmitter_prediction_v2.csv",
                        usecols=["root_id"] + sorted(BB.FAST_NT))
    votes["root_id"] = votes.root_id.astype(np.int64)
    votes = votes.drop_duplicates("root_id").set_index("root_id")
    fast_cols = sorted(BB.FAST_NT)
    vsub = votes.reindex(n.root_id)[fast_cols].fillna(0).to_numpy()
    fast_vote = np.array(fast_cols, dtype=object)[vsub.argmax(1)]
    has_votes = vsub.sum(1) > 0
    ver = n.neurotransmitter_verified.astype("string").str.split(",").str[0].str.lower()
    ver_fast = ver.isin(BB.FAST_NT).to_numpy()
    ver_mod = (ver.notna() & ~ver.isin(BB.FAST_NT)).to_numpy()
    nt = np.where(ver_fast, ver.to_numpy(dtype=object),
                  np.where(has_votes, fast_vote, "unknown")).astype(object)
    nt[ver_mod] = "modulator:" + ver[ver_mod].to_numpy(dtype=object)
    n["nt"] = nt.astype(str)
    n["nt_source"] = np.where(ver_fast, "verified", np.where(ver_mod, "verified_modulator",
                              np.where(has_votes, "fast_vote", "none")))
    n["is_modulator"] = ver_mod
    n["sign"] = np.where(n.nt.isin(BB.INHIBITORY_NT), -1, 1).astype(np.int8)
    n["label"] = n.cell_type.where(n.cell_type.notna(), n.super_class).astype(str)
    log(f"transmitter: verified fast {int(ver_fast.sum()):,}, verified modulator {int(ver_mod.sum()):,} "
        f"(synapses dropped), fast-vote {int((~ver_fast & ~ver_mod & has_votes).sum()):,}, "
        f"unknown {int((~ver_fast & ~ver_mod & ~has_votes).sum()):,}")

    # ---------------------------------------------------------- area: proxy calibrated on CX, then measured
    cx = np.load(CX_NPZ, allow_pickle=True)
    cx_root = cx["root_id"].astype(np.int64)
    cx_pos = {int(r): i for i, r in enumerate(cx_root)}
    mi = meta.set_index("root_id")
    have = [int(r) for r in cx_root if int(r) in mi.index]
    V = mi.loc[have, "volume_nm3"].to_numpy(float)
    L = mi.loc[have, "l2_cable_length_um"].to_numpy(float) * 1e3
    A_est = 2 * np.sqrt(np.pi * V * L)
    A_meas = cx["area_nm2"][[cx_pos[r] for r in have]]
    frag_cx = cx["is_fragment"][[cx_pos[r] for r in have]]
    ok = (~frag_cx) & np.isfinite(A_est) & (A_est > 0)
    AREA_K = float(np.median(A_meas[ok] / A_est[ok]))
    Vn = n.volume_nm3.to_numpy(float)
    Ln = n.l2_cable_length_um.to_numpy(float) * 1e3
    with np.errstate(invalid="ignore"):
        area = AREA_K * 2 * np.sqrt(np.pi * np.nan_to_num(Vn) * np.nan_to_num(Ln))
    vol_ratio = float(np.median(V[ok] / cx["volume_nm3"][[cx_pos[r] for r in have]][ok]))
    frag_thresh = 10e9 * vol_ratio
    is_frag = (~np.isfinite(Vn) | ~np.isfinite(Ln) | (Vn < frag_thresh) | ~np.isfinite(area) | (area <= 0))
    measured = np.zeros(len(n), bool)
    src = []
    for pf in (BULK / "l2_full.parquet", BULK / "l2_vnc.parquet", BULK / "l2_patch.parquet"):
        if not pf.exists():
            continue
        patch = pd.read_parquet(pf).drop_duplicates("root_id").set_index("root_id")
        m = n.root_id.isin(patch.index).to_numpy() & ~measured
        pa = patch.reindex(n.root_id[m])
        good = (pa.volume_nm3.to_numpy(float) >= 10e9) & (pa.area_nm2.to_numpy(float) > 0)
        rows = np.flatnonzero(m)[good]
        area[rows] = pa.area_nm2.to_numpy(float)[good]
        is_frag[rows] = False; measured[rows] = True
        src.append(f"{pf.name}: {int(good.sum()):,}")
    med_area = float(np.median(area[~is_frag]))
    area[is_frag] = med_area
    log(f"areas: measured {int(measured.sum()):,} ({'; '.join(src)}), proxy {int((~measured & ~is_frag).sum()):,}, "
        f"fragments/median {int(is_frag.sum()):,}  (K {AREA_K:.3f})")
    by_region = {k: int(measured[(n.region == k).to_numpy()].sum()) for k in KEEP_REGION}
    log(f"  measured by region: {by_region}")

    # cable attenuation factor (kept for completeness; unused by the balance rule)
    R_M, R_A = 20_000.0, 100.0
    Ld = n.dend_length.to_numpy(float) if "dend_length" in n else np.full(len(n), np.nan)
    Ltot = n.l2_cable_length_um.to_numpy(float)
    Ld = np.where(np.isfinite(Ld) & (Ld > 0), Ld, 0.5 * np.nan_to_num(Ltot))
    d_cm = 2 * np.sqrt(np.nan_to_num(Vn) / (np.pi * np.maximum(np.nan_to_num(Ltot) * 1e3, 1.0))) * 1e-7
    lam_um = np.sqrt(R_M * d_cm / (4 * R_A)) * 1e4
    with np.errstate(divide="ignore", invalid="ignore"):
        x = Ld / lam_um
        atten = np.where(x > 1e-6, (1 - np.exp(-x)) / x, 1.0)
    atten[~np.isfinite(atten) | is_frag] = float(np.nanmedian(atten[np.isfinite(atten) & ~is_frag]))

    # ---------------------------------------------------------- edges
    idx = pd.Series(np.arange(len(n)), index=n.root_id)
    e = edges[edges.pre.astype(np.int64).isin(idx.index) & edges.post.astype(np.int64).isin(idx.index)]
    pre_idx = idx.loc[e.pre.astype(np.int64)].to_numpy(np.int32)
    post_idx = idx.loc[e.post.astype(np.int64)].to_numpy(np.int32)
    counts = e["count"].to_numpy(np.int32)
    log(f"edges inside the set: {len(counts):,} pairs, {int(counts.sum()):,} synapses")
    is_mod = n.is_modulator.to_numpy()
    keep_e = ~is_mod[pre_idx]
    n_mod_edges, n_mod_syn = int((~keep_e).sum()), int(counts[~keep_e].sum())
    pre_idx, post_idx, counts = pre_idx[keep_e], post_idx[keep_e], counts[keep_e]
    log(f"dropped modulator synapses: {n_mod_edges:,} pairs, {n_mod_syn:,} synapses; remaining "
        f"{len(counts):,} pairs, {int(counts.sum()):,} synapses")
    reg = n.region.fillna("").to_numpy()
    cross = (reg[pre_idx] != reg[post_idx])
    log(f"  brain<->VNC edges: {int(cross.sum()):,} pairs, {int(counts[cross].sum()):,} synapses")

    # CX check
    is_core = n.root_id.isin(set(cx_root[cx["is_core"].astype(bool)].tolist())).to_numpy()
    in_cx = n.root_id.isin(set(cx_root.tolist())).to_numpy()
    Wcx = np.abs(cx["W"]); core_set = set(cx_root[cx["is_core"].astype(bool)].tolist())
    root_arr = n.root_id.to_numpy()
    checked = mism = 0
    for p, q, c in zip(root_arr[pre_idx], root_arr[post_idx], counts):
        if int(p) in core_set and int(q) in core_set:
            checked += 1; mism += int(Wcx[cx_pos[int(p)], cx_pos[int(q)]] != c)
    log(f"CX core present {int(is_core.sum())}/151; core->core edges {checked} checked, {mism} mismatches "
        f"{'OK' if mism == 0 else 'FAIL'}")
    log("by super_class: " + ", ".join(f"{k} {v:,}" for k, v in n.super_class.fillna('NA').value_counts().items()))

    np.savez_compressed(
        OUT / "banc_full.npz",
        root_id=n.root_id.to_numpy(np.int64), label=n.label.to_numpy(),
        cell_type=n.cell_type.fillna("").astype(str).to_numpy(),
        super_class=n.super_class.fillna("").astype(str).to_numpy(),
        region=n.region.fillna("").astype(str).to_numpy(),
        side=n.side.fillna("").astype(str).to_numpy(),
        nt=n.nt.to_numpy(), nt_source=n.nt_source.astype(str).to_numpy(),
        sign=n.sign.to_numpy(np.int8),
        area_nm2=area, area_measured=measured, volume_nm3=np.nan_to_num(Vn), atten=atten,
        is_fragment=is_frag, is_cx_core=is_core, in_cx_circuit=in_cx,
        pre_idx=pre_idx, post_idx=post_idx, n_syn=counts,
    )
    with open(OUT / "banc_full.json", "w") as f:
        json.dump({
            "datastack": "brain_and_nerve_cord_public", "materialization": 888,
            "neuron_set": "region∈{central_brain,ventral_nerve_cord} ∪ super_class∈{visual_projection,"
                          "visual_centrifugal} − {glia,not_a_neuron,trachea}",
            "n_neurons": int(len(n)), "n_edges": int(len(counts)), "n_synapses": int(counts.sum()),
            "by_region": {k: int(v) for k, v in n.region.fillna("NA").value_counts().items()},
            "by_super_class": {k: int(v) for k, v in n.super_class.fillna("NA").value_counts().items()},
            "brain_vnc_edges": int(cross.sum()), "brain_vnc_synapses": int(counts[cross].sum()),
            "areas": {"measured": int(measured.sum()), "measured_by_region": by_region,
                      "proxy": int((~measured & ~is_frag).sum()), "fragments": int(is_frag.sum()),
                      "K": AREA_K, "sources": src},
            "modulators_dropped": {"n_neurons": int(is_mod.sum()), "n_edges": n_mod_edges, "n_synapses": n_mod_syn},
            "cx_check": {"core_present": int(is_core.sum()), "edges_checked": checked, "mismatches": mism},
        }, f, indent=2)
    log(f"wrote {OUT / 'banc_full.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
