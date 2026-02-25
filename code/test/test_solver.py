from geometric_fv.solver import SolverState

import numpy as np
import pytest


def test_override():
    ncells = 5

    u_old = np.zeros(ncells)
    u_new = np.zeros(ncells)
    slope = np.zeros(ncells)
    cfl = 1.2

    state = SolverState(u_old=u_old, u_new=u_new, slope=slope, cfl=cfl)

    u_new_i = state.get_u_new(2)
    assert pytest.approx(u_new_i) == 0.0

    u_new_i_current = 2.1
    u_new_i = state.get_u_new(2, u_new_i_current)
    assert pytest.approx(u_new_i) == u_new_i_current
