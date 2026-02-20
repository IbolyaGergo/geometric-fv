import numpy as np
import pytest

from geometric_fv.slope import compute_slope


def test_compute_slope_Box():
    # slope = (u_old[i] - u_new[i]) / cfl

    u_old = np.array([4])
    u_new = np.array([1])

    cfl = 1.5

    i = 0
    slope = compute_slope(u_old, u_new, cfl, i)
    assert pytest.approx(slope) == 2.0
