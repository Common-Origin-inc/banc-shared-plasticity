"""
Tests of build_network(model="shiu") (the Shiu et al. 2024 formulation) on small synthetic graphs,
in Brian2 runtime mode with the numpy target. No data files, no GPU.

  a  one excitatory spike onto a resting neuron: peak depolarisation and its time against the
     closed form of dv/dt = (v_0 - v + g)/t_mbr, dg/dt = -g/tau, g(0) = w
  b  g is zero right after a postsynaptic spike with the reset, and is not without it
  c  an inhibitory edge hyperpolarises by the same closed form

run: python -m pytest BANC_BEST_BRAIN/tests/test_shiu_mode.py
"""
import sys
from pathlib import Path

import numpy as np
import pytest

BB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BB))
from brian2 import StateMonitor, mV, ms, prefs, second, seed, start_scope   # noqa: E402
import plasticity as PL                                                      # noqa: E402

prefs.codegen.target = "numpy"                           # plasticity.py asks for cython
BS = PL.BS
N = 30
N_SMALL, N_BIG = 5, 200                                  # synapses: far below threshold / enough for a spike
TM, TS = float(BS.t_mbr / ms), float(BS.tau_syn / ms)
# response of v to g(0) = w: w * PEAK at T_PEAK after the event
T_PEAK = np.log(TM / TS) * TM * TS / (TM - TS)
PEAK = TS / (TM - TS) * (np.exp(-T_PEAK / TM) - np.exp(-T_PEAK / TS))


def pair_graph():
    """0 (exc) -> 1 (N_SMALL) and -> 2 (N_BIG); 3 (inh) -> 4 (N_SMALL); the rest unconnected"""
    sign = np.ones(N, np.int8); sign[3] = -1
    return dict(n=N, sign=sign, area=np.full(N, 1e9), pre_idx=np.array([0, 0, 3]), post_idx=np.array([1, 2, 4]),
                n_syn=np.array([N_SMALL, N_BIG, N_SMALL], float), sens_idx=np.array([], np.int64))


def run_pair(g_reset, T_ms=40.0):
    """neurons 0 and 3 start above threshold, so each fires once in the first step"""
    start_scope()
    net, G, syn, spikes, masks = PL.build_network(pair_graph(), model="shiu", g_reset=g_reset, duration_s=T_ms / 1000)
    G.v[[0, 3]] = -40 * mV
    mon = StateMonitor(G, ["v", "g"], record=[1, 2, 4]); net.add(mon)
    PL.run(net, T_ms / 1000)
    return (np.asarray(mon.t / ms), np.asarray(mon.v / mV) - float(BS.v_0 / mV), np.asarray(mon.g / mV),
            np.asarray(spikes.t / ms), np.asarray(spikes.i))


def test_single_epsp_peak_matches_closed_form():
    t, dv, g, st, si = run_pair(True)
    assert list(si[si != 2]) == [0, 3] and np.all(st[si != 2] == 0.0)
    w = N_SMALL * float(BS.W_SYN_SHIU / mV)
    assert dv[0].max() == pytest.approx(w * PEAK, rel=1e-3)
    assert t[np.argmax(dv[0])] == pytest.approx(float(BS.T_DLY_SHIU / ms) + T_PEAK, abs=0.2)
    assert g[0].max() == pytest.approx(w, rel=1e-12)                # the jump itself: n_syn * 0.275 mV
    assert np.all(dv[0][t < float(BS.T_DLY_SHIU / ms)] == 0.0)      # nothing before the delay


@pytest.mark.parametrize("g_reset", [True, False])
def test_g_after_postsynaptic_spike(g_reset):
    t, dv, g, st, si = run_pair(g_reset)
    assert (si == 2).sum() >= 1
    k = int(np.flatnonzero(np.isclose(t, st[si == 2][0]))[0]) + 1   # first sample after the spike's reset
    assert dv[1][k] == 0.0                                           # v = v_rst in both modes
    if g_reset:
        assert g[1][k] == 0.0
    else:
        assert g[1][k] > 10.0                                        # mV; most of the 55 mV jump is still there


def test_inhibitory_edge_hyperpolarises():
    t, dv, g, st, si = run_pair(True)
    w = N_SMALL * float(BS.W_SYN_SHIU / mV)
    assert dv[2].max() == 0.0
    assert dv[2].min() == pytest.approx(-w * PEAK, rel=1e-3)


def test_stimulus_event_is_f_synapses():
    data = pair_graph(); data["sens_idx"] = np.array([10])
    start_scope()
    net, G, syn, spikes, masks = PL.build_network(data, model="shiu", stim=(np.array([11]), 0.0, 0.02, 7.0),
                                                  sens_hz=10.0, duration_s=0.02)
    assert float(net["drive_syn"].w_ext[0] / mV) == pytest.approx(7.0 * 0.275)
    assert float(net["sens_syn"].w_s[0] / mV) == pytest.approx(BS.SENS_KICK_F * 0.275)
    assert list(net["sens_syn"].j) == [10] and list(net["drive_syn"].j) == [11]



def test_published_stimulation_makes_the_neuron_fire_once_per_event():
    """stim_v=True: an input event jumps v by F_POI_SHIU * W_SYN_SHIU (68.75 mV), far above the 7 mV gap, so the
    stimulated neuron fires at the event rate; with the project convention at f = 3 it does not fire at all"""
    out = {}
    for sv in (False, True):
        start_scope(); seed(3)
        data = pair_graph()
        net, G, syn, spikes, _ = PL.build_network(data, model="shiu", stim=(np.array([0]), 0.0, 1.0, 3.0), duration_s=1.0, stim_v=sv)
        net.run(1.0 * second)
        out[sv] = int((np.asarray(spikes.i) == 0).sum())
    assert out[False] == 0
    assert 100 <= out[True] <= 200                           # Poisson at STIM_HZ = 150 Hz for 1 s, 2.2 ms refractory

@pytest.mark.parametrize("kw", [dict(plastic=True), dict(gains=dict(ee=np.ones(2))), dict(tau_e=15 * ms),
                                dict(escape=(1.0, 50.0)), dict(std=(0.2, 0.2)), dict(adapt=(1.0, 200.0)),
                                dict(g_syn_ps=20.0), dict(recip=0.2)])
def test_conductance_options_are_refused(kw):
    with pytest.raises(ValueError, match="shiu"):
        PL.build_network(pair_graph(), model="shiu", **kw)


def test_g_reset_off_needs_shiu_mode():
    with pytest.raises(ValueError, match="g_reset"):
        PL.build_network(pair_graph(), g_reset=False)

