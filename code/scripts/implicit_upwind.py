import matplotlib.pyplot as plt
import numpy as np

from geometric_fv.boundary import BCType, apply_bc
from geometric_fv.grid import Grid1D
from geometric_fv.schemes import Box

scheme = Box()
nghost = scheme.nghost

mesh = Grid1D.uniform(0.0, 1.0, 50)
x_c = mesh.centers
ncells = mesh.ncells

# u0 = np.sin(2*np.pi*x_c)
u0 = np.piecewise(x_c, [x_c < 0.2, (x_c >= 0.2) & (x_c < 0.5), x_c >= 0.5], [0, 1, 0])

u_new = np.zeros(ncells + 2 * nghost)
u_old = np.pad(u0, (nghost, nghost), "constant", constant_values=0.0)

bc_type = BCType.QUASI_PERIODIC
cfl = 1.6
for t in range(1):
    apply_bc(bc_type, u_old, u_new, nghost, cfl)

    scheme.sweep(u_old, u_new, cfl)
    u_old[:] = u_new[:]

    # plt.figure()
    # plt.plot(x_c, u0, x_c, u_new[1:-1])
    # plt.savefig(f'box_{str(t).zfill(3)}.png')

plt.plot(x_c, u0, "-o", x_c, u_new[1:-1], "-o")
plt.show()
