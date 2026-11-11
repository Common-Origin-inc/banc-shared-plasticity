"""Ablation: which biophysical ingredient does the bump need?

Test 5 changed two things at once relative to the failed global-w_syn Test 4:
  (a) reversal potentials  -- conductance synapses instead of voltage jumps
  (b) per-neuron leak      -- g_L,i from BANC membrane area
This runs the full 2x2 through the same build/run/analyse path.

  voltage synapse + uniform leak   = Test 4 (Shiu synapse, standard reset)
  voltage synapse + area leak      = area only
  conductance     + uniform leak   = reversal only
  conductance     + area leak      = Test 5 (the model)
"""
import json
from pathlib import Path
import numpy as np
import ring_attractor_sim as S

OUT = Path(__file__).resolve().parent
N_SEEDS = 3
SWEEP = {False: [5.0, 10.0, 20.0, 40.0, 80.0],          # conductance: pS
         True:  [0.275, 0.40, 0.55, 0.70, 1.0]}          # voltage: mV
UNIT = {False: "pS", True: "mV"}

d = np.load(OUT / "banc_cx_circuit.npz", allow_pickle=True)
W = d["W"].astype(float); lab = d["labels"].astype(str); ang = d["angles"]
area = d["area_nm2"].astype(float)
epg = np.flatnonzero(lab == "EPG")
order = epg[np.argsort(ang[epg])]
hub = epg[np.argmax(np.abs(W[np.ix_(epg, epg)]).sum(1))]
hp = int(np.flatnonzero(order == hub)[0])
seed_set = order[[(hp - S.ARC_SIZE // 2 + k) % len(order)
                  for k in range(S.ARC_SIZE)]].tolist()
t_axis = np.arange(int(round(S.T_END / (S.DT_BIN_MS / 1000)))) * S.DT_BIN_MS / 1000

cells = [("voltage synapse + uniform leak  (Test 4)", True, True),
         ("voltage synapse + AREA leak     (area only)", True, False),
         ("conductance     + uniform leak  (reversal only)", False, True),
         ("conductance     + AREA leak     (Test 5, the model)", False, False)]

summary = {}
for name, volt, unif in cells:
    print(f"=== {name}")
    rows = {}
    for x in SWEEP[volt]:
        runs = []
        for s in range(N_SEEDS):
            rr = S.run_sim(W, area, x, [(S.STIM_ON, S.STIM_OFF, seed_set, S.R_POI)],
                           S.T_END, seed_val=100 + s,
                           voltage_synapse=volt, uniform_leak=unif)
            runs.append(S.analyse(rr, epg, order, ang, t_axis))
        agg = {k: (float(np.mean([r[k] for r in runs])),
                   float(np.std([r[k] for r in runs], ddof=1)))
               for k in ("epg_hz_after", "active_hz_after", "n_active_after",
                         "R_after", "pr_after", "jaccard_+1.0", "persist_s",
                         "live_frac")}
        rows[f"{x:g} {UNIT[volt]}"] = agg
        print(f"  {x:>6g} {UNIT[volt]} | active {agg['n_active_after'][0]:4.1f} EPG @ "
              f"{agg['active_hz_after'][0]:5.0f} Hz  R {agg['R_after'][0]:.3f}  "
              f"PR {agg['pr_after'][0]:5.1f}  J+1 {agg['jaccard_+1.0'][0]:.2f}  "
              f"live {agg['live_frac'][0]:.2f}  "
              f"persist {agg['persist_s'][0]:.2f}±{agg['persist_s'][1]:.2f} s  "
              f"| {S.verdict(agg)}")
    bump = [k for k, a in rows.items()
            if a["live_frac"][0] > 0.9 and a["pr_after"][0] < 20 and a["R_after"][0] > 0.35]
    summary[name] = {"rows": rows, "persistent_localised_at": bump}
    print(f"  -> persistent localised bump at: {bump if bump else 'NONE'}\n")

print("=" * 80)
print("2x2 summary (persistent + localised, any rate):")
for name in summary:
    b = summary[name]["persistent_localised_at"]
    print(f"  {name:<52} {'YES  ' + ', '.join(b) if b else 'no'}")

with open(OUT / "experiment1_ablation.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nwrote {OUT / 'experiment1_ablation.json'}")
