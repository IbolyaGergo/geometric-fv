import pytest

from geometric_fv.equations import Burgers, LinearAdvection


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


# test_linear_advection_invert_implicit() {{{2
@pytest.mark.parametrize("a", [1.0, 2.0])
@pytest.mark.parametrize("rhs", [0.5, 1.0, -1.0])
@pytest.mark.parametrize("dt_dx", [0.1, 0.5, 1.0])
def test_linear_advection_invert_implicit(a, rhs, dt_dx):
    eq = LinearAdvection(a=a)
    res = eq.invert_implicit(rhs, dt_dx, tol=1e-9)
    u_sol = res.u

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


# test_burgers_invert_implicit() {{{2
@pytest.mark.parametrize("rhs", [0.1, 1.0, 2.0])
@pytest.mark.parametrize("dt_dx", [0.1, 0.5, 1.0])
def test_burgers_invert_implicit(rhs, dt_dx):
    eq = Burgers()
    res = eq.invert_implicit(rhs, dt_dx, tol=1e-9)
    u_sol = res.u

    assert u_sol + dt_dx * eq.flux(u_sol) == pytest.approx(rhs)


# test_burgers_invert_implicit_fallback() {{{2
def test_burgers_invert_implicit_fallback():
    eq = Burgers()
    # rhs small enough to make discriminant < 1
    # discriminant = 1 + 2 * dt/dx * rhs
    # if rhs = -1 and dt/dx = 1 => disc = -1 < 1
    rhs = -1.0
    dt_dx = 1.0
    res = eq.invert_implicit(rhs, dt_dx, tol=1e-9)

    assert res.is_invertible is False
    assert res.u == rhs


# test_burgers_invert_implicit_zero_dt_dx() {{{2
def test_burgers_invert_implicit_zero_dt_dx():
    eq = Burgers()
    rhs = 1.5
    res = eq.invert_implicit(rhs, dt_dx=0.0, tol=1e-9)

    u_sol = res.u
    assert u_sol == pytest.approx(rhs)

# test_burgers_invert_implicit_negative_sweep() {{{2
@pytest.mark.parametrize("rhs", [-0.1, -1.0])
def test_burgers_invert_implicit_negative_sweep(rhs):
    dt_dx = 1.8
    eq = Burgers()
    res = eq.invert_implicit(rhs, dt_dx, tol=1e-9, sweep_sign=-1)
    u_sol = res.u
    assert u_sol - dt_dx * eq.flux(u_sol) == pytest.approx(rhs)
