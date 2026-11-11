"""
Tests of the pure ring-bump metric functions in banc_common.py on synthetic inputs with
analytically known answers. No simulation, no data files.

banc_common imports four Brian2 units for its neuron constants; the metric functions do not use
them. Where Brian2 is not installed the units are stubbed as plain floats for the import only.

Tests named *_DIFFERS_FROM_DOC pin behaviour that departs from the documented definition
(TESTS.md / the docstring). They assert what the code does today and are not a statement that
the behaviour is right.

run: python -m pytest BANC_BEST_BRAIN/tests/test_metrics.py
"""
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import brian2  # noqa: F401
    import banc_common as BS
except ImportError:
    sys.modules["brian2"] = types.SimpleNamespace(mV=1e-3, ms=1e-3, uF=1e-6, cm=1e-2)
    try:
        import banc_common as BS
    finally:
        del sys.modules["brian2"]

N = 45                                                   # ring EPGs
ANG = np.arange(N) * 2 * np.pi / N                       # evenly spaced preferred angles
DEG = 360.0 / N                                          # 8 deg per EPG


def circ_diff(a, b):
    return np.abs(np.angle(np.exp(1j * (a - b))))


# ---- decode: population vector angle and localisation R ----

def test_decode_single_active_neuron():
    r = np.zeros((N, 1)); r[7, 0] = 30.0
    th, R = BS.decode(r, ANG)
    assert circ_diff(th[0], ANG[7]) < 1e-12 and R[0] == pytest.approx(1.0)


def test_decode_flat_profile_is_unlocalised():
    th, R = BS.decode(np.full((N, 1), 12.0), ANG)
    assert R[0] == pytest.approx(0.0, abs=1e-12)


def test_decode_silent_ring_gives_zero_not_nan():
    th, R = BS.decode(np.zeros((N, 3)), ANG)
    assert np.array_equal(R, np.zeros(3)) and np.isfinite(th).all()


def test_decode_two_opposite_neurons_cancel():
    ang = np.array([0.0, np.pi]); r = np.array([[5.0], [5.0]])
    assert BS.decode(r, ang)[1][0] == pytest.approx(0.0, abs=1e-12)


def test_decode_cosine_bump():
    # r = 1 + cos(theta - theta0) on a uniform ring: angle theta0, R = 1/2
    th0 = ANG[11]
    th, R = BS.decode((1 + np.cos(ANG - th0))[:, None], ANG)
    assert circ_diff(th[0], th0) < 1e-12 and R[0] == pytest.approx(0.5)


def test_decode_wraparound_bump():
    # bump on the last two and first three EPGs: centre is index 0, across the 0 / 2 pi seam
    r = np.zeros((N, 1)); r[[N - 2, N - 1, 0, 1, 2], 0] = 20.0
    th, R = BS.decode(r, ANG)
    assert circ_diff(th[0], ANG[0]) < 1e-12
    assert R[0] == pytest.approx(np.sin(5 * np.pi / N) / (5 * np.sin(np.pi / N)))


def test_decode_R_is_scale_invariant_and_per_bin():
    r = np.zeros((N, 2)); r[3:8, 0] = 1.0; r[3:8, 1] = 100.0
    th, R = BS.decode(r, ANG)
    assert R[0] == pytest.approx(R[1]) and th[0] == pytest.approx(th[1])


# ---- bump_fwhm_deg: contiguous EPGs at or above half maximum around the peak ----

def test_fwhm_silent_profile():
    assert BS.bump_fwhm_deg(np.zeros(N)) == 0.0


def test_fwhm_single_active_neuron_is_one_sector():
    p = np.zeros(N); p[20] = 50.0
    assert BS.bump_fwhm_deg(p) == pytest.approx(DEG)


def test_fwhm_flat_profile_is_full_ring():
    assert BS.bump_fwhm_deg(np.full(N, 9.0)) == pytest.approx(360.0)


def test_fwhm_rectangular_bump():
    p = np.zeros(N); p[10:16] = 25.0                     # six EPGs = the 45 deg Kim sector, rounded up
    assert BS.bump_fwhm_deg(p) == pytest.approx(6 * DEG)


def test_fwhm_wraparound_bump():
    p = np.zeros(N); p[[N - 2, N - 1, 0, 1]] = 25.0
    assert BS.bump_fwhm_deg(p) == pytest.approx(4 * DEG)


def test_fwhm_only_the_group_around_the_peak_counts():
    p = np.zeros(N); p[5:8] = 10.0; p[6] = 12.0; p[30:40] = 9.0   # second, wider group is not contiguous
    assert BS.bump_fwhm_deg(p) == pytest.approx(3 * DEG)


def test_fwhm_triangular_profile():
    p = np.zeros(N); p[18:25] = [2, 4.9, 8, 10, 8, 4.9, 2]       # half maximum = 5
    assert BS.bump_fwhm_deg(p) == pytest.approx(3 * DEG)


def test_fwhm_neuron_exactly_at_half_maximum_counts_DIFFERS_FROM_DOC():
    # docstring: "above half maximum"; the code counts >= half, so a neuron exactly at half is in
    p = np.zeros(N); p[18:25] = [2, 5, 8, 10, 8, 5, 2]
    assert BS.bump_fwhm_deg(p) == pytest.approx(5 * DEG)


# ---- participation_ratio: effective number of active EPGs ----

def test_participation_ratio():
    r = np.zeros((N, 3)); r[4, 0] = 17.0; r[10:19, 1] = 3.0
    assert np.allclose(BS.participation_ratio(r), [1.0, 9.0, 0.0])


# ---- jaccard_vs_reference: overlap of the active set with the set at t_ref ----

def test_jaccard_identical_disjoint_partial_and_empty():
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    r = np.zeros((6, 5))
    r[[0, 1], 0] = 10.0                                  # reference set {0, 1}
    r[[0, 1], 1] = 3.0                                   # same set, other rates -> 1
    r[[2, 3], 2] = 10.0                                  # disjoint -> 0
    r[[1, 2], 3] = 10.0                                  # one of three -> 1/3
    jac = BS.jaccard_vs_reference(r, t, 0.0)             # column 4 silent: {} vs {0, 1} -> 0
    assert np.allclose(jac, [1.0, 1.0, 0.0, 1 / 3, 0.0])


def test_jaccard_threshold_is_strict_and_empty_union_is_zero():
    t = np.array([0.0, 1.0])
    r = np.full((4, 2), BS.ACTIVE_HZ)                    # exactly at threshold = not active
    assert np.array_equal(BS.jaccard_vs_reference(r, t, 0.0), [0.0, 0.0])


def test_jaccard_reference_is_nearest_bin():
    t = np.array([0.0, 1.0, 2.0])
    r = np.zeros((3, 3)); r[0, 0] = r[1, 1] = r[2, 2] = 10.0
    assert np.array_equal(BS.jaccard_vs_reference(r, t, 1.4), [0.0, 1.0, 0.0])


# ---- bin_rates and smooth ----

def test_bin_rates_counts_and_units():
    st = np.array([0.001, 0.002, 0.012, 0.019, 0.5]); si = np.array([1, 1, 1, 0, 1])
    out = BS.bin_rates(st, si, 3, 0.02)                  # 4 bins of 5 ms; the 0.5 s spike is outside
    assert out.shape == (3, 4)
    assert np.array_equal(out[1], [400.0, 0.0, 200.0, 0.0])
    assert np.array_equal(out[0], [0.0, 0.0, 0.0, 200.0]) and not out[2].any()


def test_bin_rates_idx_subset_keeps_order():
    st = np.array([0.001, 0.006]); si = np.array([2, 0])
    out = BS.bin_rates(st, si, 3, 0.01, idx=np.array([2, 0]))
    assert np.array_equal(out, [[200.0, 0.0], [0.0, 200.0]])


def test_smooth_is_a_normalised_boxcar():
    w = int(round(BS.SMOOTH_MS / BS.DT_BIN_MS))
    r = np.zeros((2, 100)); r[0, 50] = 40.0; r[1] = 7.0
    sm = BS.smooth(r)
    assert sm[0].sum() == pytest.approx(40.0) and np.count_nonzero(sm[0]) == w
    assert sm[0].max() == pytest.approx(40.0 / w)
    assert np.allclose(sm[1, w:-w], 7.0)                 # interior of a constant row is unchanged
    assert sm[1, 0] < 7.0 and sm[1, -1] < 7.0            # edges are zero-padded, so they sag


# ---- analyse: the composite used by the scorecard (active count, rate, R, alive, persistence) ----

T_AXIS = np.arange(int(round(BS.T_END / (BS.DT_BIN_MS / 1000)))) * BS.DT_BIN_MS / 1000
ALL = np.arange(N)


def ring_rates(neurons, hz, t_on, t_off):
    r = np.zeros((N, len(T_AXIS)))
    r[np.ix_(neurons, np.flatnonzero((T_AXIS >= t_on) & (T_AXIS < t_off)))] = hz
    return r


def test_analyse_persistent_bump():
    a = BS.analyse(ring_rates(np.arange(5), 20.0, BS.STIM_ON, BS.T_END), ALL, ALL, ANG, T_AXIS)
    assert a["n_active_after"] == 5
    assert a["active_hz_after"] == pytest.approx(20.0)
    assert a["epg_hz_after"] == pytest.approx(20.0 * 5 / N)
    assert a["epg_hz_before"] == 0.0
    assert a["R_after"] == pytest.approx(np.sin(5 * np.pi / N) / (5 * np.sin(np.pi / N)))
    assert a["pr_after"] == pytest.approx(5.0)
    assert a["live_frac"] == 1.0
    assert a["jaccard_+1.0"] == 1.0 and a["jaccard_+2.0"] == 1.0
    assert a["persist_s"] == pytest.approx(BS.T_END - BS.STIM_OFF)


def test_analyse_bump_that_dies_one_second_after_light_off():
    a = BS.analyse(ring_rates(np.arange(5), 20.0, BS.STIM_ON, BS.STIM_OFF + 1.0), ALL, ALL, ANG, T_AXIS)
    dark = BS.T_END - BS.STIM_OFF - 0.05                 # length of the scored window
    assert a["n_active_after"] == 5                      # mean over the window is still > 2 Hz
    assert a["persist_s"] == pytest.approx(1.0, abs=BS.SMOOTH_MS / 1000)
    assert a["live_frac"] == pytest.approx(0.95 / dark, abs=0.02)
    assert a["jaccard_+2.0"] == 0.0


def test_analyse_silent_ring():
    a = BS.analyse(np.zeros((N, len(T_AXIS))), ALL, ALL, ANG, T_AXIS)
    assert a["n_active_after"] == 0 and a["active_hz_after"] == 0.0
    assert a["R_after"] == 0.0 and a["pr_after"] == 0.0 and a["live_frac"] == 0.0


def test_analyse_live_frac_ignores_localisation_DIFFERS_FROM_DOC():
    # TESTS.md: "alive = fraction of time bins with R > 0.5". validate_scorecard.py reports
    # analyse()["live_frac"] as "alive", and live_frac is the fraction of post-light-off bins in
    # which the summed EPG rate exceeds ACTIVE_HZ; R does not enter. A uniformly active ring
    # (no bump, R = 0) therefore scores alive = 1. capability_tests.py computes alive from R > 0.5.
    a = BS.analyse(ring_rates(ALL, 20.0, 0.0, BS.T_END), ALL, ALL, ANG, T_AXIS)
    assert a["R_after"] == pytest.approx(0.0, abs=1e-9)
    assert a["live_frac"] == 1.0
