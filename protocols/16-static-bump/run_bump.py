"""
Protocol 16: the bump in the static brain-and-cord network (silent, 5-ms synapses, no learning).
usage: run_bump.py --seed S [--device cuda_standalone]   -> results/bump_static_s<S>.json
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BB = HERE.parents[1] / "BANC_BEST_BRAIN"
sys.path.insert(0, str(BB))


def arg(flag, default, cast=str):
    a = sys.argv
    return cast(a[a.index(flag) + 1]) if flag in a else default


DEVICE = arg("--device", "runtime")
if DEVICE != "runtime":
    import brian2
    if DEVICE == "cuda_standalone":
        import brian2cuda  # noqa: F401
    brian2.set_device(DEVICE, build_on_run=False)
import plasticity as PL  # noqa: E402
from brian2 import ms, second, start_scope, seed, device  # noqa: E402
from provenance import provenance  # noqa: E402

BS = PL.BS
sd = arg("--seed", 1, int)
T, ON, OFF = BS.T_END, BS.STIM_ON, BS.STIM_OFF            # 4.0, 0.5, 1.547 s
data = PL.load_data(str(BB / "banc_full.npz")); n = data["n"]
# ring order and cue exactly as every other test (capability_tests.ring, offset 0)
cx = np.load(BB / "banc_cx_circuit.npz", allow_pickle=True)
cx_root = cx["root_id"].astype(np.int64); lab = cx["labels"].astype(str); cx_ang = cx["angles"]
e = np.flatnonzero(lab == "EPG"); order = e[np.argsort(cx_ang[e])]
pos = {int(r): i for i, r in enumerate(data["root"])}
epg = np.array([pos[int(cx_root[i])] for i in order]); ang = cx_ang[order]
stim = PL.kim_stim(data)
stim_rows = sorted(int(np.flatnonzero(epg == s)[0]) for s in stim)

if DEVICE != "runtime":
    device.reinit(); device.activate(build_on_run=False, directory=f"/tmp/brian_static_bump_s{sd}")
start_scope(); seed(sd)
net, G, syn, spikes, masks = PL.build_network(data, g_syn_ps=30.0, plastic=False, gains=None, sens_hz=0.0,
                                               stim=(stim, ON, OFF, 250.0), stim2=None, duration_s=T,
                                               tau_e=5.0 * ms, escape=None, std=None)
PL.run(net, T)
if DEVICE != "runtime":
    device.build(run=True)
st, si = np.asarray(spikes.t / second), np.asarray(spikes.i)

t_axis = np.arange(int(round(T / (BS.DT_BIN_MS / 1000)))) * BS.DT_BIN_MS / 1000
r_epg = BS.bin_rates(st, si, n, T, epg)
a = BS.analyse(r_epg, np.arange(len(epg)), np.arange(len(epg)), ang, t_axis)
after = t_axis > OFF + 0.05
per = np.bincount(si[st > OFF + 0.05], minlength=n)[epg] / (T - OFF - 0.05)
active_rows = [int(i) for i in np.flatnonzero(per > BS.ACTIVE_HZ)]
zc = np.exp(1j * ang[stim_rows]).mean(); zb = (np.exp(1j * ang) * per).sum()
res = dict(seed=sd, T=T, stim_on=ON, stim_off=OFF, stim_contacts_per_event=250.0, stim_hz=BS.STIM_HZ,
           n_active_after=a["n_active_after"], active_hz_after=a["active_hz_after"],
           fwhm_deg=float(BS.bump_fwhm_deg(r_epg[:, after].mean(1))), R_after=a["R_after"],
           live_frac=a["live_frac"], persist_s=a["persist_s"], epg_hz_before=a["epg_hz_before"],
           stim_rows=stim_rows, active_rows=active_rows,
           stim_rows_active=int(len(set(stim_rows) & set(active_rows))),
           cue_angle_deg=float(np.degrees(np.angle(zc))), bump_angle_deg=float(np.degrees(np.angle(zb))),
           n_active_brain_after=int((np.bincount(si[st > OFF + 0.05], minlength=n) / (T - OFF - 0.05) > 2).sum()),
           provenance=provenance())
(HERE / "results").mkdir(exist_ok=True)
(HERE / "results" / f"bump_static_s{sd}.json").write_text(json.dumps(res, indent=1))
print(f"[s{sd}] {res['n_active_after']} EPG @ {res['active_hz_after']:.1f} Hz, FWHM {res['fwhm_deg']:.0f} deg, "
      f"R {res['R_after']:.3f}, persist {res['persist_s']:.3f} s, stimulated rows active {res['stim_rows_active']}/6, "
      f"cue {res['cue_angle_deg']:+.0f} bump {res['bump_angle_deg']:+.0f} deg")
