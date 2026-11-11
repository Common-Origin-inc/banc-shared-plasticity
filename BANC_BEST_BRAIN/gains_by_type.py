"""What did a plasticity run do to the loop's synapses? Per pre->post type means of the saved gains
(relative to the v2 weights).
usage: gains_by_type.py w_final_TAG.npz [more.npz ...]"""
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
import plasticity as PL  # noqa: E402

data = PL.load_data(OUT / "banc_full.npz")
_, masks, _ = PL.v2_weights(data)
ct = data["cell_type"]; pre, post = data["pre_idx"], data["post_idx"]
for f in sys.argv[1:]:
    g = np.load(f)
    print(f"\n{f}: <ee> {g['ee'].mean():.3f} <ie> {g['ie'].mean():.3f}")
    for k, m in (("ee", masks["ee"]), ("ie", masks["ie"])):
        gk = g[k]; p_, q_ = pre[m], post[m]
        pairs = [("EPG", "PEN_a(PEN1)"), ("EPG", "PEN_b(PEN2)"), ("PEN_a(PEN1)", "EPG"), ("PEN_b(PEN2)", "EPG"),
                 ("EPG", "PEG"), ("PEG", "EPG"), ("KC", "KC"), ("PN", "KC")] if k == "ee" else \
                [("ExR6", "EPG"), ("ExR4", "PEN_b(PEN2)"), ("ExR6", "PEN_a(PEN1)"), ("Delta7", "EPG"),
                 ("Delta7", "PEN_b(PEN2)"), ("Delta7", "PEG"), ("ER4d", "EPG"), ("ER2_c", "EPG"),
                 ("GLNO", "PEN_a(PEN1)"), ("APL", "KC")]
        out = []
        for a, b in pairs:
            sa = np.char.startswith(ct[p_], a) if a in ("KC", "PN", "ER") else (ct[p_] == a)
            sb = np.char.startswith(ct[q_], b) if b in ("KC", "PN") else (ct[q_] == b)
            if a == "PN":
                sa = np.char.endswith(ct[p_], "PN")
            sel = sa & sb
            if sel.any():
                out.append(f"{a}->{b} {gk[sel].mean():.2f} (max {gk[sel].max():.1f}, n {int(sel.sum())})")
        print(f"  {k}: " + " | ".join(out))
