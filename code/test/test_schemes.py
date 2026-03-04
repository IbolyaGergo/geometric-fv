from dataclasses import dataclass

import numpy as np

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


# test_SecondOrderImplicit_equals_Box() {{{2
def test_SecondOrderImplicit_equals_Box():
    config = SolverConfig(
        mesh=mesh_config,
        reconst=ReconstConfig(slope_type=SlopeType.BOX, limiter_type=LimiterType.NONE),
    )
    cfl = 1.6

    scheme_2ndo = SecondOrderImplicit(config=config)
    state_2ndo = init_solver_state(scheme_2ndo, cfl)

    scheme_box = Box(config=config)
    state_box = init_solver_state(scheme_box, cfl)

    for s, st in [(scheme_2ndo, state_2ndo), (scheme_box, state_box)]:
        s.apply_bc(st)
        s.sweep(st)

    np.testing.assert_allclose(state_2ndo.u_new, state_box.u_new)


# test_SecondOrderImplicit_equals_ImplicitUpwind_when_limit_is_FULL() {{{2
def test_SecondOrderImplicit_equals_ImplicitUpwind_when_limit_is_FULL():
    config = SolverConfig(
        mesh=mesh_config,
        reconst=ReconstConfig(slope_type=SlopeType.BOX, limiter_type=LimiterType.FULL),
    )
    cfl = 1.6

    scheme_2ndo = SecondOrderImplicit(config=config)
    state_2ndo = init_solver_state(scheme_2ndo, cfl)

    scheme_box = ImplicitUpwind(config=config)
    state_box = init_solver_state(scheme_box, cfl)

    for s, st in [(scheme_2ndo, state_2ndo), (scheme_box, state_box)]:
        s.apply_bc(st)
        s.sweep(st)

    np.testing.assert_allclose(state_2ndo.u_new, state_box.u_new)
