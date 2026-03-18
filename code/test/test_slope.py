import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays as hnp

from geometric_fv.config import ReconstConfig
from geometric_fv.enums import LimiterType, SlopeType
from geometric_fv.slope import compute_slope
from geometric_fv.solver import SolverState


# TESTs {{{1
# test_compute_slope_Box() {{{2
def test_compute_slope_Box():
    # slope = (u_old[i] - u_new[i]) / dt_dx

    u_old = np.array([4])
    u_new = np.array([1])
    slope = np.zeros(len(u_old))

    dt_dx = 1.5
    i = 0

    slope_type = SlopeType.BOX
    limiter_type = LimiterType.NONE
    reconst_config = ReconstConfig(slope_type=slope_type, limiter_type=limiter_type)

    state = SolverState(u_old=u_old, u_new=u_new, slope=slope, dt_dx=dt_dx)
    slope_i = compute_slope(state, i, u_new_i=u_new[i], reconst_config=reconst_config)
    assert pytest.approx(slope_i) == 2.0

    slope_i = compute_slope(state, i, u_new_i=2.5, reconst_config=reconst_config)
    assert pytest.approx(slope_i) == 1.0


# test_compute_slope_Box_indexing() {{{2
def test_compute_slope_Box_indexing():
    # slope = (u_old[i] - u_new[i]) / dt_dx
    ncells = 20
    u_old = np.zeros(ncells)
    u_new = np.zeros(ncells)

    i = 5
    u_old[i] = 9.6
    u_new[i] = 6
    slope = np.array([0.0])

    dt_dx = 1.2

    slope_type = SlopeType.BOX
    limiter_type = LimiterType.NONE
    reconst_config = ReconstConfig(slope_type=slope_type, limiter_type=limiter_type)

    state = SolverState(u_old=u_old, u_new=u_new, slope=slope, dt_dx=dt_dx)
    slope_i = compute_slope(state, i, u_new_i=u_new[i], reconst_config=reconst_config)

    assert pytest.approx(slope_i) == 3.0


# test_tvd_suff_properties_hypothesis() {{{2
# We want arrays of length 3 (i-1, i, i+1) and positive dt_dx values
u_st = hnp(np.float64, 3, elements=st.floats(-1e6, 1e6))
dt_dx_st = st.floats(0.1, 2.0)


@given(u_old=u_st, u_new=u_st, dt_dx=dt_dx_st)
def test_limit_slope_tvd_suff_properties_hypothesis(u_old, u_new, dt_dx):
    """
    Hypothesis will automatically try edge cases: 0.0, NaN, Inf, very
    small/large differences, etc.
    """
    reconst_config = ReconstConfig(
        slope_type=SlopeType.BOX, limiter_type=LimiterType.TVD_SUFF
    )
    state = SolverState(u_old=u_old, u_new=u_new, slope=np.zeros(3), dt_dx=dt_dx)

    # Calculate theoretical bounds (The "Oracle")
    i = 1
    slope_box = (u_old[i] - u_new[i]) / dt_dx
    A = 2.0 * (u_old[i + 1] - u_new[i]) / (1.0 + dt_dx)
    B = (2.0 / dt_dx) * (u_old[i] - u_new[i - 1]) / (1.0 + dt_dx)

    # Act
    slope_lim = compute_slope(state, i, u_new[i], reconst_config)

    # Assert Invariants
    # 1. The Sign Invariant: slope must have the same sign as the box slope or be zero
    if abs(slope_lim) > 1e-14:
        assert np.sign(slope_lim) == np.sign(slope_box)

    # 2. The Minmod Invariant:
    # If any signs differ, it MUST be zero.
    # If all signs match, it MUST be bounded by the minimum absolute value.
    if (slope_box * A > 0) and (slope_box * B > 0):
        assert abs(slope_lim) <= min(abs(slope_box), abs(A), abs(B)) + 1e-12
    else:
        assert pytest.approx(slope_lim) == 0.0


# test_tvd_necessary_properties_hypothesis() {{{2
ncells = 4
u_st = hnp(np.float64, ncells, elements=st.floats(-10, 10))
slope_st = hnp(np.float64, ncells, elements=st.floats(-10, 10))
dt_dx_st = st.floats(0.1, 2.0)


@given(u_old=u_st, u_new=u_st, slope=slope_st, dt_dx=dt_dx_st)
def test_limit_slope_tvd_necessary_properties_hypothesis(u_old, u_new, slope, dt_dx):
    config_suff = ReconstConfig(
        slope_type=SlopeType.BOX, limiter_type=LimiterType.TVD_SUFF
    )
    config_nec = ReconstConfig(slope_type=SlopeType.BOX, limiter_type=LimiterType.TVD)
    state = SolverState(u_old=u_old, u_new=u_new, slope=slope, dt_dx=dt_dx)

    # Make sure slope[i-1] is properly bounded
    i = 2
    slope[i - 1] = compute_slope(state, i - 1, u_new[i - 1], config_suff)

    slope_suff = compute_slope(state, i, u_new[i], config_suff)
    slope_nec = compute_slope(state, i, u_new[i], config_nec)

    A = 2.0 * (u_old[i + 1] - u_new[i]) / (1.0 + dt_dx)
    B = (2.0 / dt_dx) * (u_old[i] - u_new[i - 1]) / (1.0 + dt_dx)

    # Downwind limit
    assert abs(slope_nec) <= abs(A) + 1e-12

    # Necessary should be less restrictive
    assert abs(slope_nec) >= abs(slope_suff) - 1e-12

    assert np.sign(B) * (-dt_dx * B + slope[i - 1]) <= np.sign(B) * slope_nec + 1e-12
    assert np.sign(B) * (B + slope[i - 1]) >= np.sign(B) * slope_nec - 1e-12
