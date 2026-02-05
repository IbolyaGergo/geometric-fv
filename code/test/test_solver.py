import numpy as np
import pytest

from geometric_fv.grid import Grid1D
from geometric_fv.solver import SchemeType, update_cell, sweep

@pytest.mark.parametrize("val", range(5))
def test_update_cell_implicit_upwind_constant_input(val):
    u_stencil_old = np.ones(3) * val
    u_stencil_new = np.ones(3) * val
    assert pytest.approx(val) == update_cell(1, u_stencil_old, u_stencil_new,
                                              1,
                                              scheme_type=SchemeType.IMPLICIT_UPWIND)

@pytest.mark.parametrize("u_old_i, u_new_im1, expected", [(1, 2, 1.5)])
def test_update_cell_implicit_upwind_concrete_values(u_old_i, u_new_im1, expected):
    u_stencil_old = np.array([0, u_old_i, 0])
    u_stencil_new = np.array([u_new_im1, 0, 0])
    assert pytest.approx(expected) == update_cell(1, u_stencil_old, u_stencil_new,
                                                   1,
                                                   scheme_type=SchemeType.IMPLICIT_UPWIND)

def test_sweep_implicit_upwind_ones():
    ncells = 10
    u_old = np.ones(ncells)
    u_new = np.ones(ncells)
    cfl = 1
    sol = sweep(ncells=ncells, u_old=u_old, cfl=cfl,
                scheme_type=SchemeType.IMPLICIT_UPWIND,
                get_bc_func=lambda u_old, cfl: u_old[:2])
    np.testing.assert_allclose(sol, np.ones(ncells))
