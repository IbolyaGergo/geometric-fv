import numpy as np
import pytest

from geometric_fv.grid import Grid1D
from geometric_fv.boundary import *

mesh = Grid1D.uniform(0.0, 1.0, 50)

x_c = mesh.centers
ncells = mesh.ncells

u0 = np.sin(2*np.pi*x_c)


@pytest.mark.parametrize("nghost", [1, 2])
def test_apply_bc_quasi_periodic_nghost_1_2(nghost):
    u_old = np.pad(u0, (nghost,nghost), 'constant', constant_values=0.0)
    u_new = np.copy(u_old)

    bc_type = BCType.QUASI_PERIODIC
    apply_bc(bc_type, u_old, u_new, nghost)

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

