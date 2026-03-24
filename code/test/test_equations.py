import pytest

from geometric_fv.equations import Burgers, LinearAdvection
from geometric_fv.solver import solve_for_u


# TEST_LINEAR_ADVECTION {{{1
# test_linear_advection_flux_and_dfdu() {{{2
@pytest.mark.parametrize("a", [1.0, -2.5, 0.5])
@pytest.mark.parametrize("u", [0.0, 1.0, -1.5, 10.0])
def test_linear_advection_flux_and_dfdu(a, u):
    eq = LinearAdvection(a=a)
    assert eq.flux(u) == pytest.approx(a * u)
    assert eq.dfdu(u) == pytest.approx(a)


# test_linear_advection_speed() {{{2
@pytest.mark.parametrize("a", [1.0, 2.0])
@pytest.mark.parametrize(("u1", "u2"), [(0.0, 1.0), (1.0, -1.0), (2.0, 2.0)])
def test_linear_advection_speed(a, u1, u2):
    eq = LinearAdvection(a=a)
    assert eq.speed(u1, u2) == pytest.approx(a)


# test_linear_advection_solve_for_u() {{{2
@pytest.mark.parametrize("a", [1.0, 2.0])
@pytest.mark.parametrize("rhs", [0.5, 1.0, -1.0])
@pytest.mark.parametrize("dt_dx", [0.1, 0.5, 1.0])
def test_linear_advection_solve_for_u(a, rhs, dt_dx):
    eq = LinearAdvection(a=a)
    u_sol = solve_for_u(eq, rhs, dt_dx)

    assert u_sol + dt_dx * eq.flux(u_sol) == pytest.approx(rhs)


# TEST_BURGERS {{{1
# test_burgers_flux_and_dfdu() {{{2
@pytest.mark.parametrize("u", [0.0, 1.0, -1.5, 2.0])
def test_burgers_flux_and_dfdu(u):
    eq = Burgers()
    assert eq.flux(u) == pytest.approx(0.5 * u**2)
    assert eq.dfdu(u) == pytest.approx(u)


# test_burgers_speed() {{{2
@pytest.mark.parametrize(
    ("u1", "u2"), [(0.0, 1.0), (1.0, 2.0), (1.0, 1.0), (-1.0, 1.0)]
)
def test_burgers_speed(u1, u2):
    eq = Burgers()
    if u1 == u2:
        expected_speed = u1
    else:
        expected_speed = 0.5 * (u1 + u2)
    assert eq.speed(u1, u2) == pytest.approx(expected_speed)


# test_burgers_solve_for_u() {{{2
@pytest.mark.parametrize("rhs", [0.1, 1.0, 2.0])
@pytest.mark.parametrize("dt_dx", [0.1, 0.5, 1.0])
def test_burgers_solve_for_u(rhs, dt_dx):
    eq = Burgers()
    u_sol = solve_for_u(eq, rhs, dt_dx)

    # Note: For Burgers, rhs must be such that 1 + 2*dt_dx*rhs >= 0 for real
    # solution
    assert u_sol + dt_dx * eq.flux(u_sol) == pytest.approx(rhs)


# test_burgers_solve_for_u_zero_dt_dx() {{{2
def test_burgers_solve_for_u_zero_dt_dx():
    eq = Burgers()
    assert solve_for_u(eq, 1.5, 0.0) == pytest.approx(1.5)
