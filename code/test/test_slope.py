import numpy as np
import pytest

from geometric_fv.slope import compute_slope, SlopeType


def test_compute_slope_Box():
    # slope = (u_old[i] - u_new[i]) / cfl

    u_old = np.array([4])
    u_new = np.array([1])

    cfl = 1.5
    i = 0

    slope = np.zeros(len(u_old))
    slope_i = compute_slope(u_old, u_new, slope, cfl, i, slope_type=SlopeType.BOX)
    assert pytest.approx(slope_i) == 2.0

    slope_i = compute_slope(u_old, u_new, slope, cfl, i, slope_type=SlopeType.BOX, u_new_i_current=2.5)
    assert pytest.approx(slope_i) == 1.0


def test_compute_slope_Box_indexing():
    # slope = (u_old[i] - u_new[i]) / cfl
    ncells = 20
    u_old = np.zeros(ncells)
    u_new = np.zeros(ncells)

    i = 5
    u_old[i] = 9.6
    u_new[i] = 6

    cfl = 1.2

    slope = np.array([0.0])
    slope_i = compute_slope(u_old, u_new, slope, cfl, i, slope_type=SlopeType.BOX)

    assert pytest.approx(slope_i) == 3.0
