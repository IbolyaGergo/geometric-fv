import numpy as np
import pytest

from geometric_fv.schemes import Box, ImplicitUpwind


@pytest.mark.parametrize("scheme", [ImplicitUpwind(), Box()])
def test_constant_solution(scheme):
    ncells = 20

    u_new = np.zeros(ncells)
    u_old = np.ones(ncells)

    cfl = 1.6
    scheme.sweep(u_old, u_new, cfl)
