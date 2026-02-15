import numpy as np
from geometric_fv.schemes import ImplicitUpwind, Box
from geometric_fv.boundary import BCType
import pytest

@pytest.mark.parametrize("scheme", [
    ImplicitUpwind(),
    Box()
])
def test_constant_solution(scheme):
    ncells = 20

    u_new = np.zeros(ncells)
    u_old = np.ones(ncells)

    cfl = 1.6
    scheme.sweep(u_old, u_new, cfl)

