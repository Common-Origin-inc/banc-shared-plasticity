"""
Protocol 02 summary: one row per learned network, every primary outcome as measured.
No threshold is applied. Missing files are reported as missing, never skipped.

usage: summarize.py            -> RESULTS.md and results.csv next to this file
Reads the published checkpoint's archived results from BANC_BEST_BRAIN/results/ and each
arm's outputs from out_sub/results and out_first/results.
"""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARCH = HERE.parents[1] / "BANC_BEST_BRAIN" / "results"
STD = "stdrecip_eeU0.2t0.2"
PERSIST_SEEDS = (1, 2, 3, 4, 5)

COLS = [
    ("rest_mean_hz", "non-driven mean rate (Hz), capability rest seed 1"),
    ("rest_ceiling_pct", "non-driven neurons above 200 Hz (%)"),
    ("epg_hz", "window-active EPG rate (Hz), persistence seeds 1 to 5: min / median / max"),
    ("alive", "fraction of bins with R > 0.5, same seeds: min / median / max"),
    ("fwhm_deg", "bump width (deg), same seeds: min / median / max"),
    ("hz_bump", "bump-neuron rate (Hz), same seeds: min / median / max"),
    ("kc_da1_pct", "Kenyon cells active, DA1 at 3 synapse equivalents (%)"),
    ("gf_hz", "giant fibre rate, LC4+LPLC2 (Hz)"),
    ("mn9_sugar_hz", "MN9 rate, sugar (Hz)"),
    ("sc_rest_mean_hz", "scorecard rest mean (Hz): sensory-only background, 2 s, seed 3"),
]


def load(p):
    return json.load(open(p)) if p.exists() else None


def spread(vals):
    v = sorted(x for x in vals if x is not None)
    if not v:
        return "missing"
    med = v[len(v) // 2] if len(v) % 2 else 0.5 * (v[len(v) // 2 - 1] + v[len(v) // 2])
    miss = f" ({len(vals) - len(v)} missing)" if len(v) < len(vals) else ""
    return f"{v[0]:.2f} / {med:.2f} / {v[-1]:.2f}{miss}"


def num(x, scale=1.0, fmt="{:.2f}"):
    return "missing" if x is None else fmt.format(scale * x)


def row(label, scorecard, cap_dir, cap_suffix):
    """cap_suffix: the part of the capability file name after 'cap_p37a_<test>_s<N>_'"""
    sc = load(scorecard) or {}
    rest = load(cap_dir / f"cap_p37a_rest_s1_{cap_suffix}.json") or {}
    per = [load(cap_dir / f"cap_p37a_persist_s{s}_{cap_suffix}.json") or {} for s in PERSIST_SEEDS]
    get = lambda d, *ks: (get(d.get(ks[0]) or {}, *ks[1:]) if len(ks) > 1 else d.get(ks[0])) if d else None
    return {
        "network": label,
        "rest_mean_hz": num(rest.get("mean_hz")),
        "rest_ceiling_pct": num(rest.get("ceiling"), 100),
        "epg_hz": spread([p.get("hz_mean") for p in per]),
        "alive": spread([p.get("alive") for p in per]),
        "fwhm_deg": spread([p.get("fwhm_mean") for p in per]),
        "hz_bump": spread([p.get("hz_bump") for p in per]),
        "kc_da1_pct": num(get(sc, "olfaction", "DA1_f3", "kc_frac"), 100),
        "gf_hz": num(get(sc, "looming", "LC4+LPLC2", "gf")),
        "mn9_sugar_hz": num(get(sc, "feeding", "sugar", "mn9")),
        "sc_rest_mean_hz": num(get(sc, "rest", "mean")),
    }


CHECKS = ['## Checks', '', '- Stage 1 with seed 7 reproduces the published `w_final_p8.npz` exactly (`ee` and `ie` arrays identical).', '- Stage 1 checkpoints are identical across the two tables for every replicate.', '- Replicates 2 to 5 of both tables were also run on separate GPUs of the same model. For the same seeds,', '  stage-2 gains differed between machines by at most 4.4e-16 in any gain (last-bit rounding) in seven of', '  eight comparisons and were identical in one; every primary outcome was identical.', '- `ablate_check.py` exits nonzero when it prints FAIL under its historical thresholds; those verdicts are', '  descriptive, not errors.']


def main():
    rows = [row("published (stage 1 seed 7, stage 2 seed 11)",
                ARCH / "scorecards" / f"scorecard_p37a_{STD}_full_seed1.json", ARCH / "capability", STD)]
    for arm, sfx, table in (("sub", "", "committed table"), ("first", "f", "first-token table")):
        res = HERE / f"out_{arm}" / "results"
        for k in (1, 2, 3, 4, 5):
            r = f"rep{k}{sfx}"
            rows.append(row(f"{table}, replicate {k} (seeds {100 + k} / {200 + k})",
                            res / f"scorecard_{r}_p37a_{STD}_full_seed1.json", res, f"{STD}_g{r}_p37a"))
    keys = ["network"] + [c for c, _ in COLS]
    with open(HERE / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    md = ["# Protocol 02 results", "",
          "Measured values, one row per learned network. No threshold is applied.", "",
          "| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    md += ["| " + " | ".join(str(r[k]) for k in keys) + " |" for r in rows]
    md += ["", "## Columns", ""] + [f"- `{c}`: {d}" for c, d in COLS]
    md += [""] + CHECKS
    (HERE / "RESULTS.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[4:7 + len(rows)]))


if __name__ == "__main__":
    main()
