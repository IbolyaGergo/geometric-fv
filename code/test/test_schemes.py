import numpy as np
import pytest

from geometric_fv.schemes import ImplicitUpwind, SecondOrderImplicit
from geometric_fv.slope import SlopeType
from geometric_fv.solver import SolverState


@pytest.mark.parametrize("scheme", [ImplicitUpwind(), SecondOrderImplicit()])
def test_constant_solution(scheme):
    ncells = 20

    for val in np.linspace(0.0, 2.0, 10):
        u_new = val * np.ones(ncells)
        u_old = val * np.ones(ncells)

        cfl = 1.6

        slope = np.zeros_like(u_old)
        state = SolverState(u_old=u_old, u_new=u_new, slope=slope, cfl=cfl)
        scheme.sweep(state)

        expected = val * np.ones(ncells)
        np.testing.assert_allclose(u_new, expected)


def test_SecondOrderImplicit_equals_Box():
    ncells = 20
    cfl = 1.6

    x_c = np.linspace(0.0, 1.0, ncells)
    u_old = np.sin(2 * np.pi * x_c)
    slope = np.zeros_like(u_old)

    scheme = SecondOrderImplicit(slope_type=SlopeType.BOX)
    u_new_2ndO = np.zeros(len(u_old))
    state2ndO = SolverState(u_old=u_old, u_new=u_new_2ndO, slope=slope, cfl=cfl)
    scheme.sweep(state2ndO)

    # Box scheme
    nghost = 1
    coeff = (1 - cfl) / (1 + cfl)
    u_new_Box = np.zeros(len(u_old))
    for i in range(nghost, len(u_old) - nghost):
        u_new_Box[i] = coeff * u_old[i] + u_old[i - 1] - coeff * u_new_Box[i - 1]

    np.testing.assert_allclose(u_new_2ndO, u_new_Box)


def test_SecondOrderImplicit_equals_ImplicitUpwind_when_limit_is_ZERO():
    ncells = 20

    x_c = np.linspace(0.0, 1.0, ncells)
    u_old = np.sin(2 * np.pi * x_c)
    slope = np.zeros_like(u_old)

    cfl = 1.6
    scheme = SecondOrderImplicit(slope_type=SlopeType.ZERO)
    u_new_2ndO = np.zeros(len(u_old))
    state2ndO = SolverState(u_old=u_old, u_new=u_new_2ndO, slope=slope, cfl=cfl)
    scheme.sweep(state2ndO)

    scheme = ImplicitUpwind()
    u_new_ImplUp = np.zeros(ncells)
    stateImplUp = SolverState(u_old=u_old, u_new=u_new_ImplUp, slope=slope, cfl=cfl)
    scheme.sweep(stateImplUp)

    np.testing.assert_allclose(u_new_2ndO, u_new_ImplUp)
