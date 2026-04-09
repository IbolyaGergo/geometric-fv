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
from geometric_fv.equations import Burgers
from geometric_fv.schemes import (
    HighResImplicit,
)

# Mesh
x_min = 0.0
x_max = 1.5
ncells = 100

bc_type = BCType.CONSTANT_EXTEND
slope_type = SlopeType.BOX
limiter_type = LimiterType.TVD
guess_type = GuessType.BOX

dt_dx = 1.8

config = SolverConfig(
    mesh=MeshConfig(x_min=x_min, x_max=x_max, ncells=ncells),
    boundary=BoundaryConfig(bc_type=bc_type),
    reconst=ReconstConfig(
        slope_type=slope_type, limiter_type=limiter_type, guess_type=guess_type
    ),
    iteration=IterationConfig(tol=1e-6, maxiter=50),
    equation=Burgers(),
    dt_dx=dt_dx,
)

scheme = HighResImplicit(config=config)
nghost = scheme.nghost

mesh = config.mesh.create_mesh()
x_c = mesh.centers
ncells = mesh.ncells

# u0 = 0.5 * np.tanh((x_c-0.5)*20)+0.5
# u0 = np.sin(2 * np.pi * x_c)
# u0 = (np.sin(2 * np.pi * x_c) + 1.1) / 2.1
# u0 = np.piecewise(
#     x_c,
#     [x_c < 0.2, (x_c >= 0.2) & (x_c < 0.5), x_c >= 0.5],
#     [-0.5, 1, -0.5]
# )
u0 = np.piecewise(
    x_c, [x_c < 0.2, (x_c >= 0.2) & (x_c < 0.7), x_c >= 0.7], [0.2, 1, 0.2]
)
# u0 = np.piecewise(x_c, [x_c < 0.5, x_c >= 0.5], [1, 0])
# u0 = np.piecewise(x_c, [x_c < 0.5, x_c >= 0.5], [0, 1.0])
# u0 = 0.05 + 0.95 * np.e**(-50*(x_c - 0.5)**2)

state = scheme.allocate_state(u0)

for _t in range(10):
    scheme.apply_bc(state)
    scheme.sweep(state)

    state.u_old[:] = state.u_new[:]

    plt.figure()
    plt.plot(x_c, u0, "-o", x_c, state.u_new[nghost:-nghost], "-o")
    plt.savefig(f"{str(_t).zfill(3)}.png")
    plt.close()

# plt.plot(x_c, u0, "-o", x_c, u_new[1:-1], "-o")
# plt.show()
