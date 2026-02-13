import numpy as np
import pytest

from geometric_fv.grid import Grid1D
from geometric_fv.boundary import *

mesh = Grid1D.uniform(0.0, 1.0, 50)

x_c = mesh.centers
ncells = mesh.ncells

u0 = np.sin(2*np.pi*x_c)

bc_type = BCType.QUASI_PERIODIC

@pytest.mark.parametrize("nghost", [1, 2])
def test_apply_bc_quasi_periodic_nghost_1_2(nghost):
    u_old = np.pad(u0, (nghost,nghost), 'constant', constant_values=0.0)
    u_new = np.copy(u_old)

    apply_bc(bc_type, u_old, u_new, nghost, cfl=0.0)

    if nghost == 1:
        # 0 \\ 1 \ 2 \ ... \ -2 \\ -1
        assert u_old[0] == u_old[-1-nghost]
        assert u_old[-1] == u_old[1]
    elif nghost == 2:
        # 0 \ 1 \\ 2 \ 3 \ ... \ -4 \ -3 \\ -2 \ -1
        assert u_old[0] == u_old[-4]
        assert u_old[1] == u_old[-3]
        assert u_old[-2] == u_old[2]
        assert u_old[-1] == u_old[3]

@pytest.mark.parametrize("cfl", [0.6, 1.5, 2.7])
def test_apply_bc_quasi_periodic_cfl(cfl):
    nghost = 1
    # 0 \\ 1 \ 2 \ ... \ -2 \\ -1

    u_old = np.pad(u0, (nghost,nghost), 'constant', constant_values=0.0)
    u_new = np.copy(u_old)

    apply_bc(bc_type, u_old, u_new, nghost, cfl)
    if (cfl >= 0.0 and cfl <= 1.0):
        assert pytest.approx(u_new[0]) == (1 - cfl) * u_old[-2] + cfl * u_old[-3]
    elif (cfl >= 1.0 and cfl <= 2.0):
        assert pytest.approx(u_new[0]) == (2 - cfl) * u_old[-3] + (cfl - 1) * u_old[-4]
    elif (cfl >= 2.0 and cfl <= 3.0):
        assert pytest.approx(u_new[0]) == (3 - cfl) * u_old[-4] + (cfl - 2) * u_old[-5]

@pytest.mark.parametrize("cfl", [-0.6, -1.5, -2.7])
def test_apply_bc_quasi_periodic_negative_cfl(cfl):
    nghost = 1
    # 0 \\ 1 \ 2 \ ... \ -2 \\ -1

    u_old = np.pad(u0, (nghost,nghost), 'constant', constant_values=0.0)
    u_new = np.copy(u_old)

    apply_bc(bc_type, u_old, u_new, nghost, cfl)
    if (-cfl >= 0.0 and -cfl <= 1.0):
        assert pytest.approx(u_new[-1]) == (1 - (-cfl)) * u_old[1] + (-cfl) * u_old[2]
    if (-cfl >= 1.0 and -cfl <= 2.0):
        assert pytest.approx(u_new[-1]) == (2 - (-cfl)) * u_old[2] + (-cfl - 1) * u_old[3]
    if (-cfl >= 2.0 and -cfl <= 3.0):
        assert pytest.approx(u_new[-1]) == (3 - (-cfl)) * u_old[3] + (-cfl - 2) * u_old[4]
