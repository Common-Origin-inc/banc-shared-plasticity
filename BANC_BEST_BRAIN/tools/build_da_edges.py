"""
Dopaminergic (modulatory) edges for the three-factor rule: the synapses of verified-dopamine
neurons onto neurons of banc_full.npz, which the conductance model dropped (not fast
transmitters). Written in banc_full index space.
usage: build_da_edges.py  -> da_edges.npz (pre_idx, post_idx, n_syn, da_in per neuron) + summary
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1]                        # BANC_BEST_BRAIN/
BULK = Path(os.environ.get("BANC_BULK", OUT.parent / "bulk"))     # BANC tables, default bulk/ at the repository root

meta = pd.read_feather(BULK / "banc_888_meta.feather")
edges = pd.read_feather(BULK / "banc_888_edgelist_simple_v3.feather")
meta["root_id"] = meta.root_888.astype(np.int64)
ver = meta.neurotransmitter_verified.astype("string").str.lower().fillna("")
is_da = ver.str.contains("dopamine").to_numpy()
da_roots = set(meta.root_id[is_da].astype(int))

d = np.load(OUT / "banc_full.npz", allow_pickle=True)
sim_root = d["root_id"].astype(np.int64); n = len(sim_root)
pos = {int(r): i for i, r in enumerate(sim_root)}
ct = d["cell_type"].astype(str)

e = edges[edges.pre.astype(np.int64).isin(da_roots)]
pre = e.pre.astype(np.int64).map(pos); post = e.post.astype(np.int64).map(pos)
ok = pre.notna() & post.notna()
pre_idx = pre[ok].to_numpy().astype(np.int64); post_idx = post[ok].to_numpy().astype(np.int64)
n_syn = e["count"][ok].to_numpy(float)
da_in = np.bincount(post_idx, weights=n_syn, minlength=n)
is_dan = np.zeros(n, bool); is_dan[np.unique(pre_idx)] = True
np.savez_compressed(OUT / "da_edges.npz", pre_idx=pre_idx, post_idx=post_idx, n_syn=n_syn, da_in=da_in, is_dan=is_dan)
print(f"DA neurons in the sim: {int(is_dan.sum()):,} (of {len(da_roots):,} verified DA in BANC); "
      f"DA edges {len(pre_idx):,}, synapses {int(n_syn.sum()):,}; targets {int((da_in > 0).sum()):,} neurons")
for t in ("EPG", "PEN_a(PEN1)", "PEN_b(PEN2)", "PEG", "Delta7", "ExR6", "PFL3", "ER4d", "APL"):
    m = ct == t
    print(f"  {t:12s} DA syn/cell {da_in[m].mean():6.1f}")
kc = np.char.startswith(ct, "KC"); print(f"  {'KC':12s} DA syn/cell {da_in[kc].mean():6.1f}")
print(f"  all neurons: median {np.median(da_in):.1f}, mean {da_in.mean():.1f}, 90th pct {np.percentile(da_in, 90):.1f}")
