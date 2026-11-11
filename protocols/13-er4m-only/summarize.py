"""Protocol 13 summary: committed, first-token and ER4m-removed networks with the same seeds.
Measured values, no threshold. usage: summarize.py -> RESULTS.md"""
import glob
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent; P02 = HERE.parent / "02-replication"; S = "stdrecip_eeU0.2t0.2"


def row(label, res, r):
    sc = Path(res) / f"scorecard_{r}_p37a_{S}_full_seed1.json"
    if not sc.exists():
        return f"| {label} | missing | missing | missing | missing | missing | missing |"
    s = json.load(open(sc)); per = [json.load(open(p)) for p in sorted(glob.glob(str(Path(res) / f"cap_p37a_persist_s[1-5]_{S}_g{r}_p37a.json")))]
    rest = json.load(open(glob.glob(str(Path(res) / f"cap_p37a_rest_s1_{S}_g{r}_p37a.json"))[0]))
    f = lambda k, fm: f"{fm.format(min(p[k] for p in per))} to {fm.format(max(p[k] for p in per))}"
    return (f"| {label} | {rest['mean_hz']:.2f} | {f('hz_mean', '{:.1f}')} | {f('alive', '{:.2f}')} | {f('fwhm_mean', '{:.0f}')} | "
            f"{100 * s['olfaction']['DA1_f3']['kc_frac']:.1f} | {s['looming']['LC4+LPLC2']['gf']:.1f} |")


md = ["# Protocol 13 results", "", "Same seeds in the three arms. Ranges are over the five 20 s evaluation seeds.", "",
      "| network | rest (Hz) | EPG rate (Hz) | alive | width (deg) | KC, DA1 (%) | GF (Hz) |", "|---|---|---|---|---|---|---|"]
for k in range(1, 6):
    md.append(row(f"replicate {k}, committed table", P02 / "out_sub/results", f"rep{k}"))
    md.append(row(f"replicate {k}, first-token table (ten sources removed)", P02 / "out_first/results", f"rep{k}f"))
    md.append(row(f"replicate {k}, four ER4m sources removed", HERE / f"k{k}/out_noer4m/results", f"rep{k}e"))
md += ["", "One RTX 4090 GPU per replicate."]
(HERE / "RESULTS.md").write_text("\n".join(md) + "\n"); print("\n".join(md[4:]))
