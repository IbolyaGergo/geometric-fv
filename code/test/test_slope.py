import numpy as np
import pytest

from geometric_fv.config import ReconstConfig
from geometric_fv.enums import LimiterType, SlopeType
from geometric_fv.slope import compute_slope
from geometric_fv.solver import SolverState


def test_compute_slope_Box():
    # slope = (u_old[i] - u_new[i]) / cfl

    u_old = np.array([4])
    u_new = np.array([1])
    slope = np.zeros(len(u_old))

    cfl = 1.5
    i = 0

    slope_type = SlopeType.BOX
    limiter_type = LimiterType.NONE
    config = ReconstConfig(slope_type=slope_type, limiter_type=limiter_type)

    state = SolverState(u_old=u_old, u_new=u_new, slope=slope, cfl=cfl)
    slope_i = compute_slope(state, i, u_new_i=u_new[i], config=config)
    assert pytest.approx(slope_i) == 2.0

    slope_i = compute_slope(state, i, u_new_i=2.5, config=config)
    assert pytest.approx(slope_i) == 1.0


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
    config = ReconstConfig(slope_type=slope_type, limiter_type=limiter_type)

    state = SolverState(u_old=u_old, u_new=u_new, slope=slope, cfl=cfl)
    slope_i = compute_slope(
        state, i, u_new_i=u_new[i], config=config
    )

    assert pytest.approx(slope_i) == 3.0
