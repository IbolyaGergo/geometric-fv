import numpy as np
import pytest

from geometric_fv.config import ReconstConfig
from geometric_fv.enums import LimiterType, SlopeType
from geometric_fv.slope import compute_slope
from geometric_fv.solver import SolverState


# TESTs {{{1
# test_compute_slope_Box() {{{2
def test_compute_slope_Box():
    # slope = (u_old[i] - u_new[i]) / cfl

    u_old = np.array([4])
    u_new = np.array([1])
    slope = np.zeros(len(u_old))

    cfl = 1.5
    i = 0

    slope_type = SlopeType.BOX
    limiter_type = LimiterType.NONE
    reconst_config = ReconstConfig(slope_type=slope_type, limiter_type=limiter_type)

    state = SolverState(u_old=u_old, u_new=u_new, slope=slope, cfl=cfl)
    slope_i = compute_slope(state, i, u_new_i=u_new[i], reconst_config=reconst_config)
    assert pytest.approx(slope_i) == 2.0

    slope_i = compute_slope(state, i, u_new_i=2.5, reconst_config=reconst_config)
    assert pytest.approx(slope_i) == 1.0


# test_compute_slope_Box_indexing() {{{2
def test_compute_slope_Box_indexing():
    # slope = (u_old[i] - u_new[i]) / cfl
    ncells = 20
    u_old = np.zeros(ncells)
    u_new = np.zeros(ncells)

    i = 5
    u_old[i] = 9.6
    u_new[i] = 6
    slope = np.array([0.0])

    cfl = 1.2

    slope_type = SlopeType.BOX
    limiter_type = LimiterType.NONE
    reconst_config = ReconstConfig(slope_type=slope_type, limiter_type=limiter_type)

    state = SolverState(u_old=u_old, u_new=u_new, slope=slope, cfl=cfl)
    slope_i = compute_slope(state, i, u_new_i=u_new[i], reconst_config=reconst_config)

    assert pytest.approx(slope_i) == 3.0


# test_limit_slope_tvd_suff_random() {{{2
def test_limit_slope_tvd_suff_random():
    """Property-based test for TVD_SUFF limiter."""
    reconst_config = ReconstConfig(
        slope_type=SlopeType.BOX, limiter_type=LimiterType.TVD_SUFF
    )

    ncells = 20
    np.random.seed(42)
    nsamples = 20
    for _ in range(nsamples):
        cfl = np.random.uniform(0.1, 2.0)
        u_old = np.random.uniform(-1.0, 1.0, ncells)
        u_new = np.random.uniform(-1.0, 1.0, ncells)
        slope = np.random.uniform(-1.0, 1.0, ncells)

        slope[0] = 0.0
        for i in np.arange(1, ncells - 1):
            state = SolverState(u_old=u_old, u_new=u_new, slope=slope, cfl=cfl)

            # BOX slope: slope_i = (u_old[i] - u_new[i]) / cfl
            slope_i = (u_old[i] - u_new[i]) / cfl

            # Limit bounds:
            # Sufficient condition:
            A = 2.0 * (u_old[i + 1] - u_new[i]) / (1.0 + cfl)
            B = (2.0 / cfl) * (u_old[i] - u_new[i - 1]) / (1.0 + cfl)
            # Necessary condition
            C = -cfl * B + slope[i - 1]
            D = B + slope[i - 1]

            slope_i_lim = compute_slope(
                state=state, i=i, u_new_i=u_new[i], reconst_config=reconst_config
            )

            # Check boundedness: abs(slope_i_lim) <= min(abs(A), abs(B))
            if slope_i * A > 0 and slope_i * B > 0:
                assert abs(slope_i_lim) <= abs(slope_i) + 1e-12
                assert abs(slope_i_lim) <= abs(A) + 1e-12
                assert abs(slope_i_lim) <= abs(B) + 1e-12
                assert abs(slope_i_lim) <= abs(D) + 1e-12
                # Check it's exactly the minmod
                expected = np.sign(slope_i) * min(abs(slope_i), abs(A), abs(B))
                assert pytest.approx(slope_i_lim) == expected

                # Necessary condition
                assert np.sign(B) * C <= 1e-12
                assert np.sign(B) * C <= np.sign(B) * slope_i_lim
            else:
                assert pytest.approx(slope_i_lim) == 0.0

            slope[i] = slope_i_lim
