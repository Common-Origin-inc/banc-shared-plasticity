"""
Pass/fail for one ablation of the best brain (used by cap_run.sh ablate to decide whether to run
the full suite on it).  usage: ablate_check.py NAME CAP_TAG_SUFFIX SCORECARD_JSON
  CAP_TAG_SUFFIX: the capability tag after "p37a_persist_sN_", e.g. stdrecip_eeU0.2t0.2_gnone
Criteria (the reference brain: bump 22-33 Hz, alive 0.87-0.90, rest 4.8 Hz, KC 3.4 %, GF 16 Hz):
  bump   : all 3 seeds alive >= 0.75 and bump-EPG rate 8-45 Hz, ring peak < 100 Hz
  brain  : rest mean < 12 Hz and ceiling < 1 %; DA1 x3 KC 1-15 %; looming GF > 5 Hz
prints one line "ABLATION NAME: PASS|FAIL | ..." and exits 0 on PASS, 1 on FAIL.
"""
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
name, suf, scf = sys.argv[1], sys.argv[2], sys.argv[3]
notes, ok = [], True
rates, alive = [], []
for s in (1, 2, 3):
    p = OUT / "results_cap" / f"cap_p37a_persist_s{s}_{suf}.json"
    if not p.exists():
        notes.append(f"seed {s} missing"); ok = False; continue
    j = json.load(open(p)); rates.append(j.get("hz_bump", 0.0)); alive.append(j.get("alive", 0.0))
    if j.get("alive", 0) < 0.75 or not (8 <= j.get("hz_bump", 0) <= 45) or j.get("hz_ring_max", 0) >= 100:
        ok = False
notes.append("bump " + " ".join(f"{r:.0f}Hz/{a:.2f}" for r, a in zip(rates, alive)))
try:
    sc = json.load(open(scf))
    r = sc["rest"]; notes.append(f"rest {r['mean']:.1f}Hz ceil {100*r['ceiling']:.1f}%")
    if r["mean"] >= 12 or r["ceiling"] >= 0.01: ok = False
    kc = sc["olfaction"]["DA1_f3"]["kc_frac"]; notes.append(f"KC {100*kc:.1f}%")
    if not (0.01 <= kc <= 0.15): ok = False
    gf = sc["looming"]["LC4+LPLC2"]["gf"]; notes.append(f"GF {gf:.0f}Hz")
    if gf <= 5: ok = False
except Exception as e:  # noqa: BLE001
    notes.append(f"scorecard unreadable ({e})"); ok = False
print(f"ABLATION {name}: {'PASS' if ok else 'FAIL'} | " + " | ".join(notes), flush=True)
sys.exit(0 if ok else 1)
