import numpy as np
import pytest

from geometric_fv.schemes import Box, ImplicitUpwind, SecondOrderImplicit


@pytest.mark.parametrize("scheme", [ImplicitUpwind(), Box(), SecondOrderImplicit()])
def test_constant_solution(scheme):
    ncells = 20

    for val in np.linspace(0.0, 2.0, 10):
        u_new = val * np.ones(ncells)
        u_old = val * np.ones(ncells)

        cfl = 1.6
        scheme.sweep(u_old, u_new, cfl)

        expected = val * np.ones(ncells)
        np.testing.assert_allclose(u_new, expected)


def test_SecondOrderImplicit_equals_Box():
    ncells = 20
    cfl = 1.6

    x_c = np.linspace(0.0, 1.0, ncells)
    u_old = np.sin(2 * np.pi * x_c)

    scheme = SecondOrderImplicit()
    u_new_2ndO = np.zeros(len(u_old))
    scheme.sweep(u_old, u_new_2ndO, cfl)

    scheme = Box()
    u_new_Box = np.zeros(len(u_old))
    scheme.sweep(u_old, u_new_Box, cfl)

    np.testing.assert_allclose(u_new_2ndO, u_new_Box)
