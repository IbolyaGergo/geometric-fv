import matplotlib.pyplot as plt
import numpy as np

from geometric_fv.config import (
    BoundaryConfig,
    IterationConfig,
    MeshConfig,
    ReconstConfig,
    SolverConfig,
)
from geometric_fv.enums import BCType, GuessType, LimiterType, SlopeType
from geometric_fv.schemes import SecondOrderImplicit

# Mesh
x_min = 0.0
x_max = 1.0
ncells = 100

bc_type = BCType.QUASI_PERIODIC
slope_type = SlopeType.BOX
limiter_type = LimiterType.TVD
guess_type = GuessType.BOX

config = SolverConfig(
    mesh=MeshConfig(x_min=x_min, x_max=x_max, ncells=ncells),
    boundary=BoundaryConfig(bc_type=bc_type),
    reconst=ReconstConfig(
        slope_type=slope_type, limiter_type=limiter_type, guess_type=guess_type
    ),
    iteration=IterationConfig(tol=1e-6, maxiter=50),
)

scheme = SecondOrderImplicit(config=config)
nghost = scheme.nghost

mesh = config.mesh.create_mesh()
x_c = mesh.centers
ncells = mesh.ncells

u0 = np.sin(2 * np.pi * x_c)
# u0 = np.e**(-(x_c - 0.5)**2*100)
# u0 = np.piecewise(x_c, [x_c < 0.2, (x_c >= 0.2) & (x_c < 0.5), x_c >= 0.5], [0, 1, 0])

cfl = 0.9

state = scheme.allocate_state(u0, cfl=cfl)
state_neg = scheme.allocate_state(u0, cfl=-cfl)

for _t in range(100):
    scheme.apply_bc(state)
    scheme.sweep(state)

    scheme.apply_bc(state_neg)
    scheme.sweep(state_neg)

    state.u_old[:] = state.u_new[:]
    state_neg.u_old[:] = state_neg.u_new[:]

    plt.figure()
    plt.plot(x_c, 2 * u0, "-o", x_c, state.u_new[nghost:-nghost] + state_neg.u_new[nghost:-nghost], "-o")
    plt.savefig(f"tvd_box_{str(_t).zfill(3)}.png")
    plt.close()

# plt.plot(x_c, u0, "-o", x_c, u_new[1:-1], "-o")
# plt.show()
