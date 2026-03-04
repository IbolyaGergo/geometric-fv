import numpy as np

from geometric_fv.config import BoundaryConfig, ReconstConfig, SolverConfig
from geometric_fv.boundary import apply_bc
from geometric_fv.schemes import SecondOrderImplicit
from geometric_fv.enums import BCType, LimiterType, SlopeType
from geometric_fv.solver import SolverState


def test_constant_solution():
    ncells = 20
    scheme = SecondOrderImplicit()

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
    coeff = (1 - cfl) / (1 + cfl)

    x_c = np.linspace(0.0, 1.0, ncells)
    u_old = np.sin(2 * np.pi * x_c)
    u_new_2ndO = np.zeros(len(u_old))
    slope = np.zeros_like(u_old)

    bc_type = BCType.QUASI_PERIODIC
    slope_type = SlopeType.BOX
    limiter_type = LimiterType.NONE

    config = SolverConfig(
        boundary=BoundaryConfig(bc_type=bc_type),
        reconst=ReconstConfig(slope_type=slope_type, limiter_type=limiter_type),
    )
    scheme = SecondOrderImplicit(config=config)
    nghost = scheme.nghost

    state2ndO = SolverState(u_old=u_old, u_new=u_new_2ndO, slope=slope, cfl=cfl)
    apply_bc(state2ndO, nghost=nghost, config=config.boundary,
             reconst_config=config.reconst)
    scheme.sweep(state2ndO)

    # Box scheme
    u_new_Box = np.zeros(len(u_old))
    stateBox = SolverState(u_old=u_old, u_new=u_new_Box, slope=slope, cfl=cfl)
    apply_bc(stateBox, nghost=nghost, config=config.boundary,
             reconst_config=config.reconst)
    for i in range(nghost, len(u_old) - nghost):
        u_new_Box[i] = coeff * u_old[i] + u_old[i - 1] - coeff * u_new_Box[i - 1]

    np.testing.assert_allclose(u_new_2ndO, u_new_Box)


def test_SecondOrderImplicit_equals_ImplicitUpwind_when_limit_is_FULL():
    ncells = 20
    cfl = 1.6

    x_c = np.linspace(0.0, 1.0, ncells)
    u_old = np.sin(2 * np.pi * x_c)
    slope = np.zeros_like(u_old)

    bc_type = BCType.QUASI_PERIODIC
    slope_type = SlopeType.BOX
    limiter_type = LimiterType.FULL

    config = SolverConfig(
        boundary=BoundaryConfig(bc_type=bc_type),
        reconst=ReconstConfig(slope_type=slope_type, limiter_type=limiter_type),
    )
    scheme = SecondOrderImplicit(config=config)
    nghost = scheme.nghost

    u_new_2ndO = np.zeros(len(u_old))
    state2ndO = SolverState(u_old=u_old, u_new=u_new_2ndO, slope=slope, cfl=cfl)
    apply_bc(state2ndO, nghost=nghost, config=config.boundary,
             reconst_config=config.reconst)
    scheme.sweep(state2ndO)

    # Implicit Uwpind
    u_new_ImplUp = np.zeros(ncells)
    stateImplUp = SolverState(u_old=u_old, u_new=u_new_ImplUp, slope=slope, cfl=cfl)
    apply_bc(stateImplUp, nghost=nghost, config=config.boundary,
             reconst_config=config.reconst)
    for i in range(nghost, len(u_old) - nghost):
        u_new_ImplUp[i] = (u_old[i] + cfl * u_new_ImplUp[i - 1]) / (1.0 + cfl)

    np.testing.assert_allclose(u_new_2ndO, u_new_ImplUp)
