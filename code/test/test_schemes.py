from dataclasses import dataclass

import numpy as np
import pytest

from geometric_fv.config import BoundaryConfig, MeshConfig, ReconstConfig, SolverConfig
from geometric_fv.enums import BCType, LimiterType, SlopeType
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


def sine_wave(x):
    return np.sin(2 * np.pi * x)


# TESTs {{{1
# test_constant_solution() {{{2
@pytest.mark.parametrize("val", np.linspace(0.0, 2.0, 5))
def test_constant_solution(val):
    config = SolverConfig(
        mesh=MeshConfig(ncells=20),
        boundary=BoundaryConfig(bc_type=BCType.CONSTANT_EXTEND),
    )
    scheme = SecondOrderImplicit(config=config)

    state = scheme.init_state(lambda x: np.full_like(x, val), cfl=1.6)

    scheme.apply_bc(state)
    scheme.sweep(state)

    # Verify for inner cells
    nghost = scheme.nghost
    inner_solution = state.u_new[nghost:-nghost]

    np.testing.assert_allclose(inner_solution, val, err_msg=f"Failed for val={val}")


# test_SecondOrderImplicit_equals_other_scheme_for_given_limiter() {{{2
@pytest.mark.parametrize(
    ("scheme_other", "limiter_type"),
    [(ImplicitUpwind, LimiterType.FULL), (Box, LimiterType.NONE)],
)
def test_SecondOrderImplicit_equals_other_scheme_for_given_limiter(
    scheme_other, limiter_type
):
    config = SolverConfig(
        mesh=MeshConfig(ncells=20),
        reconst=ReconstConfig(slope_type=SlopeType.BOX, limiter_type=limiter_type),
    )
    cfl = 1.6

    scheme_2ndo = SecondOrderImplicit(config=config)
    state_2ndo = scheme_2ndo.init_state(sine_wave, cfl=cfl)

    scheme_other = scheme_other(config=config)
    state_other = scheme_other.init_state(sine_wave, cfl=cfl)

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
        mesh=MeshConfig(ncells=20),
        reconst=ReconstConfig(slope_type=SlopeType.BOX, limiter_type=LimiterType.NONE),
    )
    scheme = SecondOrderImplicit(config=config)

    # Initialize state with some non-trivial data
    state = scheme.init_state(sine_wave, cfl=1.2)
    state.niter = np.zeros_like(state.u_new, dtype=int)

    scheme.apply_bc(state)
    scheme.sweep(state)

    # Check active cells (excluding ghost cells)
    ng = scheme.nghost
    active_niter = state.niter[ng:-ng]

    assert np.all(active_niter == 1), f"Expected 1 iteration, got {active_niter}"


# test_cell_indices() {{{2
def test_cell_indices():
    """
    Verifies that the cell_indices method correctly identifies the internal
    cells (excluding ghost cells) for both forward and reverse traversals.
    """

    nghost = 2
    scheme = SecondOrderImplicit(nghost=nghost)

    ninner = 20
    u0 = np.zeros(ninner)
    state = scheme.allocate_state(u0, cfl=1.0)

    assert len(state.u_old) == 24

    indices = list(scheme.cell_indices(state))
    expected_forward = list(range(nghost, len(state.u_old) - nghost))

    assert indices == expected_forward, f"Expected {expected_forward}, got {indices}"
    assert len(indices) == ninner

    rev_indices = list(scheme.cell_indices(state, reverse=True))
    expected_reverse = list(reversed(expected_forward))

    assert rev_indices == expected_reverse, (
        f"Expected {expected_reverse}, got {rev_indices}"
    )
    assert len(rev_indices) == ninner
