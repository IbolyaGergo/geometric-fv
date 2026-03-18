from dataclasses import dataclass

import numpy as np
import pytest

from geometric_fv.config import BoundaryConfig, MeshConfig, ReconstConfig, SolverConfig
from geometric_fv.enums import BCType, GuessType, LimiterType, SlopeType
from geometric_fv.schemes import Scheme, SecondOrderImplicit, HighResImplicit
from geometric_fv.solver import SolverState
from geometric_fv.equations import Burgers, LinearAdvection


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
        eq = self.config.equation

        u_old = state.u_old
        u_new = state.u_new
        dt_dx = state.dt_dx

        for i in self.cell_indices(state):
            # For positive dt_dx: u_i + dt_dx*f(u_i) = u_old_i + dt_dx*f(u_im1)
            rhs = u_old[i] + dt_dx * eq.flux(u_new[i-1])
            u_new[i] = eq.solve_for_u(rhs, dt_dx)

# Box(Scheme) {{{2
@dataclass(frozen=True)
class Box(Scheme):
    """
    A simple reference implementation of the Box scheme for testing.
    """

    nghost: int = 2
    config: SolverConfig = SolverConfig()

    def sweep(self, state: SolverState):
        dt_dx = state.dt_dx
        u_old = state.u_old
        u_new = state.u_new
        nghost = self.nghost

        coeff = (1 - dt_dx) / (1 + dt_dx)
        for i in range(nghost, len(u_old) - nghost):
            u_new[i] = coeff * (u_old[i] - u_new[i - 1]) + u_old[i - 1]


# functions {{{2
def sine_wave(x):
    return np.sin(2 * np.pi * x)
def abs_sine_wave(x):
    return abs(np.sin(2 * np.pi * x))


# TESTs {{{1
# test_constant_solution() {{{2
@pytest.mark.parametrize("val", np.linspace(0.0, 2.0, 5))
def test_constant_solution(val):
    config = SolverConfig(
        mesh=MeshConfig(ncells=20),
        boundary=BoundaryConfig(bc_type=BCType.CONSTANT_EXTEND),
    )
    scheme = SecondOrderImplicit(config=config)

    state = scheme.init_state(lambda x: np.full_like(x, val), dt_dx=1.6)

    scheme.apply_bc(state)
    scheme.sweep(state)

    # Verify for inner cells
    nghost = scheme.nghost
    inner_solution = state.u_new[nghost:-nghost]

    np.testing.assert_allclose(inner_solution, val, err_msg=f"Failed for val={val}")


# test_HighResImplicit_equals_other_scheme_for_given_limiter() {{{2
@pytest.mark.parametrize("equation", [LinearAdvection(a=1.0), Burgers()])
@pytest.mark.parametrize(
    ("scheme_other", "limiter_type"),
    [(ImplicitUpwind, LimiterType.FULL), (Box, LimiterType.NONE)],
)
def test_HighResImplicit_equals_other_scheme_for_given_limiter(
    equation, scheme_other, limiter_type
):
    """
    Verifies that HighResImplicit correctly collapses to 1st-order, or Box
    scheme (in case of LinearAdvection) when limiter is FULL or NONE
    respectively.
    """
    if scheme_other == Box and not isinstance(equation, LinearAdvection):
        pytest.skip("Box reference is only implemented for Linear Advection")

    config = SolverConfig(
        equation=equation,
        mesh=MeshConfig(ncells=20),
        reconst=ReconstConfig(limiter_type=limiter_type),
    )
    dt_dx = 1.6

    # abs_sine_wave, because Burgers works only for a > 0 yet
    scheme_hr = HighResImplicit(config=config)
    state_hr = scheme_hr.init_state(abs_sine_wave, dt_dx=dt_dx)

    scheme_other = scheme_other(config=config)
    state_other = scheme_other.init_state(abs_sine_wave, dt_dx=dt_dx)

    for s, st in [(scheme_hr, state_hr), (scheme_other, state_other)]:
        s.apply_bc(st)
        s.sweep(st)

    np.testing.assert_allclose(state_hr.u_new, state_other.u_new)


# test_iteration_count_for_exact_guess() {{{2
@pytest.mark.parametrize("dt_dx", [1.2, -1.2])
@pytest.mark.parametrize(
    ("limiter_type", "guess_type"),
    [(LimiterType.NONE, GuessType.BOX), (LimiterType.FULL, GuessType.IMPLICIT_UPWIND)],
)
def test_iteration_count_for_exact_guess(limiter_type, guess_type, dt_dx):
    """
    For SlopeType.BOX and LimiterType.NONE, the initial guess corresponds to the
    solution of the BOX scheme, thus should converge in exactly 1 iteration.
    """
    config = SolverConfig(
        mesh=MeshConfig(ncells=20),
        reconst=ReconstConfig(
            slope_type=SlopeType.BOX,
            limiter_type=limiter_type,
            guess_type=guess_type,
        ),
        boundary=BoundaryConfig(bc_type=BCType.QUASI_PERIODIC),
    )
    scheme = SecondOrderImplicit(config=config)

    # Initialize state with some non-trivial data
    state = scheme.init_state(sine_wave, dt_dx=dt_dx)
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
    state = scheme.allocate_state(u0, dt_dx=1.0)

    assert len(state.u_old) == 24

    indices = list(scheme.cell_indices(state))
    expected_forward = list(range(nghost, len(state.u_old) - nghost))

    assert indices == expected_forward, f"Expected {expected_forward}, got {indices}"
    assert len(indices) == ninner

    state_neg = scheme.allocate_state(u0, dt_dx=-1.0)

    rev_indices = list(scheme.cell_indices(state_neg))
    expected_reverse = list(reversed(expected_forward))

    assert rev_indices == expected_reverse, (
        f"Expected {expected_reverse}, got {rev_indices}"
    )
    assert len(rev_indices) == ninner


# test_mirroring() {{{2
@pytest.mark.parametrize(
    ("limiter_type", "guess_type"),
    [
        (LimiterType.NONE, GuessType.BOX),
        (LimiterType.FULL, GuessType.IMPLICIT_UPWIND),
        (LimiterType.TVD_SUFF, GuessType.BOX),
        (LimiterType.TVD, GuessType.BOX),
    ],
)
def test_mirroring(limiter_type, guess_type):
    """
    Verifies that the Implicit Upwind scheme (SecondOrderImplicit with FULL
    limiter) produces mirrored results for mirrored initial conditions and
    opposite dt_dx.
    """
    ncells = 20
    config = SolverConfig(
        mesh=MeshConfig(ncells=ncells),
        reconst=ReconstConfig(
            slope_type=SlopeType.BOX,
            limiter_type=limiter_type,
            guess_type=guess_type,
        ),
        boundary=BoundaryConfig(bc_type=BCType.QUASI_PERIODIC),
    )
    dt_dx = 1.5

    # Positive dt_dx
    scheme_pos = SecondOrderImplicit(config=config)
    state_pos = scheme_pos.init_state(sine_wave, dt_dx=dt_dx)

    scheme_pos.apply_bc(state_pos)
    scheme_pos.sweep(state_pos)

    # Negative dt_dx
    scheme_neg = SecondOrderImplicit(config=config)
    state_neg = scheme_neg.init_state(sine_wave, dt_dx=-dt_dx)

    # Manually mirror u_old from the positive case
    nghost = scheme_neg.nghost
    state_neg.u_old[nghost:-nghost] = state_pos.u_old[nghost:-nghost][::-1]

    scheme_neg.apply_bc(state_neg)
    scheme_neg.sweep(state_neg)

    # Validation
    pos_inner = state_pos.u_new[nghost:-nghost]
    neg_inner = state_neg.u_new[nghost:-nghost]

    np.testing.assert_allclose(pos_inner, neg_inner[::-1], atol=1e-12)
