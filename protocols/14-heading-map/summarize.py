"""Protocol 14 summary: persistence at eight cue positions, three seeds, six networks.
Measured values, no threshold beyond the descriptive counts fixed in PROTOCOL.md
(localization > 0.5; last-window angle within 30 deg of the cue).
usage: summarize.py -> RESULTS.md, summary.json"""
import glob
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent; S = "stdrecip_eeU0.2t0.2"
CUES = (0, 45, 90, 135, 180, 225, 270, 315); NETS = ("P", "1", "2", "3", "4", "5")
wrap = lambda a: (a + 180.0) % 360.0 - 180.0


def files(net, c, s):
    cs = f"_c{c}" if c else ""
    pat = f"cap_p37a_persist{cs}_s{s}_{S}.json" if net == "P" else f"cap_p37a_persist{cs}_s{s}_{S}_grep{net}_p37a.json"
    hits = glob.glob(str(HERE / net / "results" / pat))
    return json.load(open(hits[0])) if hits else None


def cue_angles():
    """Cue angle (deg) per offset, recomputed from the ring exactly as capability_tests.ring() selects the
    cue: in the code used for these runs, the `cue_angle` field read the radian ring angles as degrees."""
    import numpy as np
    cx = np.load(HERE.parents[1] / "BANC_BEST_BRAIN" / "banc_cx_circuit.npz", allow_pickle=True)
    lab, a = cx["labels"].astype(str), cx["angles"]; e = np.flatnonzero(lab == "EPG"); order = e[np.argsort(a[e])]
    W = cx["W"].astype(float); hub = e[np.argmax(np.abs(W[np.ix_(e, e)]).sum(1))]
    hp0 = int(np.flatnonzero(order == hub)[0]); k = int(round(45 / 360 * len(order))); out = {}
    for c in CUES:
        hp = (hp0 + int(round(c / 360 * len(order)))) % len(order)
        seed = order[[(hp - k // 2 + j) % len(order) for j in range(k)]]
        out[c] = float(np.degrees(np.angle(np.exp(1j * a[seed]).mean())))
    return out


CUE_ANG = cue_angles()


def circ_mean(angles):
    x = sum(math.cos(math.radians(a)) for a in angles); y = sum(math.sin(math.radians(a)) for a in angles)
    return math.degrees(math.atan2(y, x)), math.hypot(x, y) / max(len(angles), 1)


summary, md = {}, ["# Protocol 14 results", "",
                   "Persistence (20 s) after a cue at eight positions of the ring, three evaluation seeds each.",
                   "Cells: median localization over seeds / median signed angle of the bump in the last 2-s window",
                   "relative to the cue angle (deg). A dash: no trial with localization above 0.5.", ""]
md += ["| network | " + " | ".join(f"cue {c}" for c in CUES) + " | trials alive | alive and within 30 deg of cue | last-window angle, circular mean (resultant) |",
       "|---" * (len(CUES) + 4) + "|"]
for net in NETS:
    cells, n_alive, n_held, lasts, trials = [], 0, 0, [], []
    for c in CUES:
        rs = [r for r in (files(net, c, s) for s in (1, 2, 3)) if r is not None]
        if not rs:
            cells.append("missing"); continue
        al = sorted(r["alive"] for r in rs); errs = []
        for r in rs:
            last = r["windows"][-1]["angle"]; err = wrap(last - CUE_ANG[c])
            first = wrap(r["windows"][0]["angle"] - CUE_ANG[c])
            trials.append(dict(cue=c, seed=r["seed"], cue_angle=CUE_ANG[c], alive=r["alive"], err_first=first, err_last=err,
                               last_angle=last, n_epg=[w["n_epg"] for w in r["windows"]], hz_mean=r["hz_mean"]))
            if r["alive"] > 0.5:
                n_alive += 1; errs.append(err); lasts.append(last)
                n_held += abs(err) <= 30
        med_err = sorted(errs)[len(errs) // 2] if errs else None
        cells.append(f"{al[len(al) // 2]:.2f} / " + (f"{med_err:+.0f}" if med_err is not None else "-"))
    cm, R = circ_mean(lasts) if lasts else (float("nan"), float("nan"))
    md.append(f"| {net} | " + " | ".join(cells) + f" | {n_alive} of {len(trials)} | {n_held} of {n_alive} | {cm:+.0f} ({R:.2f}) |")
    summary[net] = dict(trials=trials, n_alive=n_alive, n_held=n_held, last_circ_mean=cm, last_resultant=R)
md += ["", "Spontaneous bump (no cue, protocol 02 / archived 8-s trials, seeds 1 and 2) against the preferred position "
       "(circular mean of the last-window angles above), and distance of the preferred position from the training sector "
       f"(cue offset 0, {CUE_ANG[0]:+.0f} deg):", "",
       "| network | spontaneous angle, seeds 1, 2 (deg) | preferred position (deg) | spontaneous minus preferred (deg) | preferred minus training sector (deg) |",
       "|---|---|---|---|---|"]
P02 = HERE.parent / "02-replication" / "out_sub" / "results"; ARCH = HERE.parents[1] / "BANC_BEST_BRAIN" / "results" / "capability"
for net in NETS:
    sp = []
    for s in (1, 2):
        f = (ARCH / f"cap_p37a_spont_s{s}_{S}.json") if net == "P" else (P02 / f"cap_p37a_spont_s{s}_{S}_grep{net}_p37a.json")
        if f.exists():
            sp.append(json.load(open(f))["windows"][0]["angle"])
    pref = summary[net]["last_circ_mean"]
    summary[net].update(spont_angles=sp, pref_minus_training=wrap(pref - CUE_ANG[0]))
    md.append(f"| {net} | " + ", ".join(f"{a:+.0f}" for a in sp) + f" | {pref:+.0f} | "
              + ", ".join(f"{wrap(a - pref):+.0f}" for a in sp) + f" | {wrap(pref - CUE_ANG[0]):+.0f} |")
md += ["", "Resultant near 1 with many cues alive: bumps end at one position whatever the cue (a preferred position).",
       "Cue angles are the circular mean of the six stimulated EPG neurons' ring angles, recomputed here: "
       + ", ".join(f"offset {c}: {CUE_ANG[c]:+.0f}" for c in CUES) + " deg. The `cue_angle` field in the result "
       "files of these runs is wrong (radians read as degrees) and is not used."]
(HERE / "RESULTS.md").write_text("\n".join(md) + "\n"); (HERE / "summary.json").write_text(json.dumps(summary, indent=1))
print("\n".join(md))
