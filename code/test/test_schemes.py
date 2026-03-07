from dataclasses import dataclass

import numpy as np
import pytest

from geometric_fv.config import MeshConfig, ReconstConfig, SolverConfig
from geometric_fv.enums import LimiterType, SlopeType
from geometric_fv.mesh import Mesh1D
from geometric_fv.schemes import Scheme, SecondOrderImplicit
from geometric_fv.solver import SolverState


# SETUP {{{1
# ImplicitUpwind(Scheme) {{{2
@dataclass(frozen=True)
class ImplicitUpwind(Scheme):
    """
    A simple reference implementation of the ImplicitUpwind scheme for testing.
    """

    nghost: int = 2
    config: SolverConfig = SolverConfig()

    def sweep(self, state: SolverState):
        cfl = state.cfl
        u_old = state.u_old
        u_new = state.u_new
        nghost = self.nghost

        coeff = 1.0 / (1.0 + cfl)
        for i in range(nghost, len(u_old) - nghost):
            u_new[i] = coeff * (u_old[i] + cfl * u_new[i - 1])


# Box(Scheme) {{{2
@dataclass(frozen=True)
class Box(Scheme):
    """
    A simple reference implementation of the Box scheme for testing.
    """

    nghost: int = 2
    config: SolverConfig = SolverConfig()

    def sweep(self, state: SolverState):
        cfl = state.cfl
        u_old = state.u_old
        u_new = state.u_new
        nghost = self.nghost

        coeff = (1 - cfl) / (1 + cfl)
        for i in range(nghost, len(u_old) - nghost):
            u_new[i] = coeff * (u_old[i] - u_new[i - 1]) + u_old[i - 1]


mesh_config = MeshConfig(x_min=0.0, x_max=1.0, ncells=20)


# init_solver_state() {{{2
def init_solver_state(scheme, cfl):
    mesh = Mesh1D.uniform(scheme.config.mesh)
    u0 = np.sin(2 * np.pi * mesh.centers)

    nghost = scheme.nghost
    u_old = np.pad(u0, (nghost, nghost), "constant", constant_values=0.0)
    u_new = np.copy(u_old)
    slope = np.zeros_like(u_old)

    return SolverState(u_old=u_old, u_new=u_new, slope=slope, cfl=cfl)


# TESTs {{{1
# test_constant_solution() {{{2
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


# test_SecondOrderImplicit_equals_other_scheme_for_given_limiter() {{{2
@pytest.mark.parametrize(
    ("scheme_other", "limiter_type"),
    [(ImplicitUpwind, LimiterType.FULL), (Box, LimiterType.NONE)],
)
def test_SecondOrderImplicit_equals_other_scheme_for_given_limiter(
    scheme_other, limiter_type
):
    config = SolverConfig(
        mesh=mesh_config,
        reconst=ReconstConfig(slope_type=SlopeType.BOX, limiter_type=limiter_type),
    )
    cfl = 1.6

    scheme_2ndo = SecondOrderImplicit(config=config)
    state_2ndo = init_solver_state(scheme_2ndo, cfl)

    scheme_other = scheme_other(config=config)
    state_other = init_solver_state(scheme_other, cfl)

    for s, st in [(scheme_2ndo, state_2ndo), (scheme_other, state_other)]:
        s.apply_bc(st)
        s.sweep(st)

    np.testing.assert_allclose(state_2ndo.u_new, state_other.u_new)


# test_iteration_count_for_box_none() {{{2
def test_iteration_count_for_box_none():
    """
    For SlopeType.BOX and LimiterType.NONE, the initial guess corresponds to the
    solution of the BOX scheme, thus should converge in exactly 1 iteration.
    """
    config = SolverConfig(
        reconst=ReconstConfig(slope_type=SlopeType.BOX, limiter_type=LimiterType.NONE)
    )
    scheme = SecondOrderImplicit(config=config)

    # Initialize state with some non-trivial data
    state = init_solver_state(scheme, cfl=1.2)
    state.niter = np.zeros_like(state.u_new, dtype=int)

    scheme.apply_bc(state)
    scheme.sweep(state)

    # Check active cells (excluding ghost cells)
    ng = scheme.nghost
    active_niter = state.niter[ng:-ng]

    assert np.all(active_niter == 1), f"Expected 1 iteration, got {active_niter}"
