import numpy as np

from geometric_fv.config import BoundaryConfig
from geometric_fv.enums import BCType
from geometric_fv.solver import SolverState


def _apply_bc_constant_extend(state: SolverState, nghost: int) -> None:
    u_old = state.u_old
    u_new = state.u_new

    # first/last idx of the physical domain
    first = nghost
    last = -nghost - 1

    for i in range(nghost):
        u_old[i] = u_old[first]
        u_new[i] = u_old[first]
        u_old[-1 - i] = u_old[last]
        u_new[-1 - i] = u_old[last]

# nghost = 1
# 0 \\ 1 \ 2 \ ... \ -2 \\ -1
# for i in [0]:
#     u[0] = u[-2]
#     u[-1] = u[1]
# nghost = 2
# 0 \ 1 \\ 2 \ 3 \ ... \ -3 \\ -2 \ -1
# for i in [0, 1]:
#     u[0] = u[-4]
#     u[1] = u[-3]
#     u[-1] = u[3]
#     u[-2] = u[2]
def _apply_bc_quasi_periodic(state: SolverState, nghost: int) -> None:
    u_old = state.u_old
    u_new = state.u_new
    cfl = state.cfl

    for i in range(nghost):
        u_old[i] = u_old[-2 * nghost + i]
        u_old[-1 - i] = u_old[2 * nghost - 1 - i]

    # fmt: off
    if cfl > 0.0:
        u_new[nghost - 1] = \
                (1 + int(np.floor(cfl)) - cfl) * u_old[-2 - int(np.floor(cfl))] \
                + (cfl - int(np.floor(cfl))) * u_old[-3 - int(np.floor(cfl))]
    else:
        u_new[-1] = \
                (1 + int(np.floor(-cfl)) - (-cfl)) * u_old[ 1 + int(np.floor(-cfl))] \
                + (-cfl - int(np.floor(-cfl))) * u_old[2 + int(np.floor(-cfl))]

    # fmt: on


_apply_bc_types = {
    BCType.CONSTANT_EXTEND: _apply_bc_constant_extend,
    BCType.QUASI_PERIODIC: _apply_bc_quasi_periodic,
}


def apply_bc(state: SolverState, config: BoundaryConfig, nghost: int) -> None:
    bc_type = config.bc_type
    apply_bc_func = _apply_bc_types.get(bc_type)
    if apply_bc_func is None:
        raise ValueError(f"Unsupported BC type: {bc_type}")
    apply_bc_func(state, nghost)

    u_old = state.u_old
    u_new = state.u_new
    slope = state.slope
    cfl = state.cfl

    if np.not_equal(cfl, 0.0):
        slope[0] = (u_old[0] - u_new[0]) / cfl
