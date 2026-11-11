"""
Secondary readouts for protocol 02, written AFTER the primary results were seen (2026-10-01).
Descriptive only. Reads the same files as summarize.py. usage: secondary.py -> SECONDARY.md
"""
import glob
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARCH = HERE.parents[1] / "BANC_BEST_BRAIN" / "results"
STD = "stdrecip_eeU0.2t0.2"


def row(label, sc_path, odors_path):
    s = json.load(open(sc_path)); o = json.load(open(odors_path))
    g = {k: f"{100 * v['kc_frac']:.1f}" for k, v in s["olf4"].items() if isinstance(v, dict) and "kc_frac" in v}
    f = s["feeding"]["sugar"]
    return (f"| {label} | {100 * o['kc_A']:.1f} / {100 * o['kc_B']:.1f} | {o['jaccard']:.2f} | "
            f"{g.get('DA1')} / {g.get('DM1')} / {g.get('VA1v')} / {g.get('DL3')} | "
            f"{s['pen']['left']:+.0f} / {s['pen']['right']:+.0f} | {f['active']:,} / {f['after']:,} | {f['kc']:.2f} |")


rows = [row("published", ARCH / "scorecards" / f"scorecard_p37a_{STD}_full_seed1.json",
            ARCH / "capability" / f"cap_p37a_odors_s1_silent_{STD}.json")]
for arm, sfx, table in (("sub", "", "committed table"), ("first", "f", "first-token table")):
    for k in (1, 2, 3, 4, 5):
        r = f"rep{k}{sfx}"; res = HERE / f"out_{arm}" / "results"
        rows.append(row(f"{table}, replicate {k}", res / f"scorecard_{r}_p37a_{STD}_full_seed1.json",
                        glob.glob(str(res / f"cap_p37a_odors_s1_silent_*g{r}_p37a.json"))[0]))
md = ["# Protocol 02 secondary readouts", "",
      "Added after the primary results were seen. Descriptive only, no threshold applied.", "",
      "| network | KC active, odor A / B (%) | two-odor Jaccard | KC active, DA1 / DM1 / VA1v / DL3 (%) | "
      "PEN shift left / right (deg) | neurons active during / after sugar | KC rate during sugar (Hz) |",
      "|---|---|---|---|---|---|---|"] + rows
(HERE / "SECONDARY.md").write_text("\n".join(md) + "\n")
print("\n".join(md[4:]))
