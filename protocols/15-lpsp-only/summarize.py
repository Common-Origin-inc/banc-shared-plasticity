"""Protocol 15 summary: committed, ER4m-removed and LPsP-removed networks with the same seeds.
Measured values, no threshold. Medians are over the five 20 s evaluation seeds, ranges in brackets.
usage: summarize.py -> RESULTS.md"""
import glob
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent; P = HERE.parent; S = "stdrecip_eeU0.2t0.2"


def row(label, res, r):
    sc = Path(res) / f"scorecard_{r}_p37a_{S}_full_seed1.json"
    if not sc.exists():
        return f"| {label} | missing | missing | missing | missing | missing | missing |"
    s = json.load(open(sc)); per = [json.load(open(p)) for p in sorted(glob.glob(str(Path(res) / f"cap_p37a_persist_s[1-5]_{S}_g{r}_p37a.json")))]
    rest = json.load(open(glob.glob(str(Path(res) / f"cap_p37a_rest_s1_{S}_g{r}_p37a.json"))[0]))
    f = lambda k, fm: (f"{fm.format(st.median(p[k] for p in per))} ({fm.format(min(p[k] for p in per))} to "
                       f"{fm.format(max(p[k] for p in per))})")
    return (f"| {label} | {rest['mean_hz']:.2f} | {f('hz_mean', '{:.2f}')} | {f('alive', '{:.2f}')} | {f('fwhm_mean', '{:.0f}')} | "
            f"{100 * s['olfaction']['DA1_f3']['kc_frac']:.1f} | {s['looming']['LC4+LPLC2']['gf']:.1f} |")


md = ["# Protocol 15 results", "",
      "Same seeds in the three arms. EPG rate, alive and width: median over the five 20 s evaluation seeds (range).", "",
      "| network | rest (Hz) | EPG rate (Hz) | alive | width (deg) | KC, DA1 (%) | GF (Hz) |", "|---|---|---|---|---|---|---|"]
for k in range(1, 6):
    md.append(row(f"replicate {k}, committed table", P / "02-replication/out_sub/results", f"rep{k}"))
    e = sorted(glob.glob(str(P / f"13-er4m-only/k{k}/**/scorecard_rep{k}e_p37a_{S}_full_seed1.json"), recursive=True))
    md.append(row(f"replicate {k}, four ER4m sources removed", Path(e[0]).parent if e else "none", f"rep{k}e"))
    l = sorted(glob.glob(str(HERE / f"k{k}/**/scorecard_rep{k}l_p37a_{S}_full_seed1.json"), recursive=True))
    md.append(row(f"replicate {k}, two LPsP sources removed", Path(l[0]).parent if l else "none", f"rep{k}l"))
(HERE / "RESULTS.md").write_text("\n".join(md) + "\n"); print("\n".join(md[4:]))
