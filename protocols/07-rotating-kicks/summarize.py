"""
Protocol 07 summary: each rotating-kick replicate next to its fixed-kick twin from protocol 02
(same seeds). Measured values only, no threshold.
usage: summarize.py -> RESULTS.md
"""
import glob
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
P02 = HERE.parent / "02-replication" / "out_sub" / "results"
STD = "stdrecip_eeU0.2t0.2"


def load(p):
    p = Path(p)
    return json.load(open(p)) if p.exists() else None


def rng(v, fmt="{:.0f}"):
    v = [x for x in v if x is not None]
    return "missing" if not v else f"{fmt.format(min(v))} to {fmt.format(max(v))}"


def row(label, res, r):
    sc = load(Path(res) / f"scorecard_{r}_p37a_{STD}_full_seed1.json")
    if sc is None:
        return f"| {label} | " + " | ".join(["missing"] * 10) + " |"
    cap = lambda t: load(glob.glob(str(Path(res) / f"cap_p37a_{t}_{STD}_g{r}_p37a.json"))[0]) if glob.glob(str(Path(res) / f"cap_p37a_{t}_{STD}_g{r}_p37a.json")) else {}
    per = [cap(f"persist_s{s}") for s in (1, 2, 3, 4, 5)]
    jumps = [cap(f"jump_o{o}_f{f}_s1") for o in (90, 180) for f in (50, 250)]
    comp = [cap(f"compete_s{s}") for s in (1, 2)]
    pen = sc["pen"]
    return (f"| {label} | {pen['left']:+.0f} / {pen['right']:+.0f} | {pen['left_unwrapped']:+.0f} / {pen['right_unwrapped']:+.0f} | "
            + " , ".join(f"{j.get('shift', float('nan')):+.0f}" for j in jumps) + " | "
            + " , ".join(str(c.get("n_bumps", "missing")) for c in comp) + " | "
            + f"{rng([p.get('hz_mean') for p in per])} | {rng([p.get('alive') for p in per], '{:.2f}')} | "
            + f"{rng([p.get('fwhm_mean') for p in per])} | {cap('rest_s1').get('mean_hz', float('nan')):.1f} | "
            + f"{100 * sc['olfaction']['DA1_f3']['kc_frac']:.1f} | {sc['looming']['LC4+LPLC2']['gf']:.1f} / {sc['feeding']['sugar']['mn9']:.1f} |")


head = ("| network | PEN shift left / right (deg) | PEN unwrapped left / right (deg) | second-cue shift: 90 deg weak, 90 strong, "
        "180 weak, 180 strong (target 90 or 180) | bumps after opposed cues, seeds 1 , 2 | EPG rate, 20 s (Hz) | alive | "
        "width (deg) | rest (Hz) | KC, DA1 (%) | GF / MN9 sugar (Hz) |")
md = ["# Protocol 07 results", "", "Rotating kicks (8 sectors) next to the fixed-kick replicate with the same seeds. "
      "Measured values, no threshold applied.", "", head, "|" + "---|" * 11]
for k in (1, 2, 3, 4, 5):
    md.append(row(f"fixed, replicate {k}", P02, f"rep{k}"))
    md.append(row(f"rotating, replicate {k}", HERE / f"k{k}" / "results", f"rep{k}r8"))
(HERE / "RESULTS.md").write_text("\n".join(md) + "\n")
print("\n".join(md[4:]))
