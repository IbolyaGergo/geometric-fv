import numpy as np
import pytest

from geometric_fv.problems import BurgersSmooth


def test_burgers_smooth_t0():
    prob = BurgersSmooth()
    x = np.linspace(0, 1, 10)
    # Check that exact at t=0 matches u0
    assert np.allclose(prob.exact(x, 0.0), prob.u0(x))


def test_burgers_smooth_residual():
    prob = BurgersSmooth()
    x = np.array([0.5])
    t = 0.2
    u = prob.exact(x, t)
    # Verify the implicit relation: u = u0(x - ut)
    u0_val = prob.u0(x - u * t)
    assert u == pytest.approx(u0_val)


def test_burgers_smooth_raises_after_shock():
    prob = BurgersSmooth()
    t_too_late = prob.t_shock + 0.1
    x = np.array([0.5])
    with pytest.raises(ValueError, match="greater than shock formation time"):
        prob.exact(x, t_too_late)
