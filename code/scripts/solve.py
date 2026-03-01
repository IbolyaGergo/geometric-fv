import matplotlib.pyplot as plt
import numpy as np

from geometric_fv.boundary import apply_bc
from geometric_fv.config import (
    BoundaryConfig,
    IterationConfig,
    ReconstConfig,
    SolverConfig,
)
from geometric_fv.enums import BCType, LimiterType, SlopeType
from geometric_fv.grid import Grid1D
from geometric_fv.schemes import SecondOrderImplicit
from geometric_fv.solver import SolverState

bc_type = BCType.QUASI_PERIODIC
slope_type = SlopeType.BOX
limiter_type = LimiterType.TVD

config = SolverConfig(
    boundary=BoundaryConfig(bc_type=bc_type),
    reconst=ReconstConfig(slope_type=slope_type, limiter_type=limiter_type),
    iteration=IterationConfig(tol=1e-6, maxiter=50),
)

scheme = SecondOrderImplicit(config=config)
nghost = scheme.nghost

mesh = Grid1D.uniform(0.0, 1.0, 100)
x_c = mesh.centers
ncells = mesh.ncells

# u0 = np.sin(2*np.pi*x_c)
u0 = np.piecewise(x_c, [x_c < 0.2, (x_c >= 0.2) & (x_c < 0.5), x_c >= 0.5], [0, 1, 0])

u_new = np.zeros(ncells + 2 * nghost)
u_old = np.pad(u0, (nghost, nghost), "constant", constant_values=0.0)
slope = np.zeros_like(u_old)

cfl = 1.6

state = SolverState(u_old=u_old, u_new=u_new, slope=slope, cfl=cfl)

for _t in range(20):
    apply_bc(state, config.boundary, nghost)

    scheme.sweep(state)

    u_old[:] = u_new[:]

    plt.figure()
    plt.plot(x_c, u0, "-o", x_c, u_new[1:-1], "-o")
    plt.savefig(f"tvd_box_{str(_t).zfill(3)}.png")

# plt.plot(x_c, u0, "-o", x_c, u_new[1:-1], "-o")
# plt.show()
