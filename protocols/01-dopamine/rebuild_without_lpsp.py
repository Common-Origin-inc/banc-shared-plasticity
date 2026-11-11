"""
Modulation table with only the two LPsP sources removed (protocol 15, the matched control for
protocol 13). LPsP is the other large modulatory input onto EPG: 2,090 of 4,573 contacts, against
2,355 for ER4m. Everything else in the committed da_edges.npz is kept. da_in and is_dan are
recomputed as in tools/build_da_edges.py. Read-only on inputs.
usage: rebuild_without_lpsp.py -> BANC_BEST_BRAIN/da_edges_nolpsp.npz
"""
from pathlib import Path

import numpy as np

BB = Path(__file__).resolve().parents[2] / "BANC_BEST_BRAIN"
g = np.load(BB / "banc_full.npz", allow_pickle=True); ct = g["cell_type"].astype(str); n = len(ct)
d = np.load(BB / "da_edges.npz", allow_pickle=True)
pre, post, n_syn = d["pre_idx"], d["post_idx"], d["n_syn"]
m = ct[pre] != "LPsP"
pre_idx, post_idx, w = pre[m], post[m], n_syn[m]
da_in = np.bincount(post_idx, weights=w, minlength=n)
is_dan = np.zeros(n, bool); is_dan[np.unique(pre_idx)] = True
np.savez_compressed(BB / "da_edges_nolpsp.npz", pre_idx=pre_idx, post_idx=post_idx, n_syn=w, da_in=da_in, is_dan=is_dan)
epg = ct == "EPG"
print(f"removed sources {len(np.unique(pre[~m]))} (cell types {sorted(set(ct[pre[~m]]))}); kept sources {int(is_dan.sum())}, "
      f"edges {len(pre_idx):,}, contacts {int(w.sum()):,}, contacts onto EPG {int(w[epg[post_idx]].sum()):,}")
