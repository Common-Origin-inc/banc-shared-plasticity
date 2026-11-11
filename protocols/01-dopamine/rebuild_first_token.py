"""
Modulation table restricted to sources whose FIRST verified transmitter token
is dopamine (selector (a) of PROTOCOL.md). Filters the committed da_edges.npz by source and
recomputes da_in and is_dan exactly as tools/build_da_edges.py does. Read-only on inputs.

usage: rebuild_first_token.py --meta PATH/banc_888_meta.feather
       -> BANC_BEST_BRAIN/da_edges_first.npz
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
BB = REPO / "BANC_BEST_BRAIN"

ap = argparse.ArgumentParser()
ap.add_argument("--meta", required=True, type=Path)
a = ap.parse_args()

meta = pd.read_feather(a.meta, columns=["root_888", "neurotransmitter_verified"])
ver = meta.neurotransmitter_verified.astype("string").str.lower().fillna("")
first = dict(zip(meta.root_888.astype(np.int64), ver.str.split(",").str[0].str.strip()))

g = np.load(BB / "banc_full.npz", allow_pickle=True)
root = g["root_id"].astype(np.int64); n = len(root); ct = g["cell_type"].astype(str)
d = np.load(BB / "da_edges.npz", allow_pickle=True)
pre, post, n_syn = d["pre_idx"], d["post_idx"], d["n_syn"]

keep_src = np.array([first.get(int(r), "") == "dopamine" for r in root])
m = keep_src[pre]
pre_idx, post_idx, w = pre[m], post[m], n_syn[m]
da_in = np.bincount(post_idx, weights=w, minlength=n)
is_dan = np.zeros(n, bool); is_dan[np.unique(pre_idx)] = True
np.savez_compressed(BB / "da_edges_first.npz", pre_idx=pre_idx, post_idx=post_idx, n_syn=w, da_in=da_in, is_dan=is_dan)
epg = ct == "EPG"
print(f"sources {int(is_dan.sum())}, edges {len(pre_idx):,}, contacts {int(w.sum()):,}, "
      f"contacts onto EPG {int(w[epg[post_idx]].sum()):,}")
