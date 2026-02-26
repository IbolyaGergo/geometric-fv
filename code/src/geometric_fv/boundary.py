from enum import Enum

import numpy as np

from geometric_fv.solver import SolverState


class BCType(Enum):
    CONSTANT_EXTEND = "constant_extend"
    QUASI_PERIODIC = "quasi_periodic"


def _apply_bc_constant_extend(state: SolverState, nghost: int) -> None:
    u_old = state.u_old
    u_new = state.u_new

    # BC
    u_old[0] = u_old[1]
    u_old[-1] = u_old[-2]
    u_new[0] = u_old[0]
    u_new[-1] = u_old[-1]


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


def apply_bc(state: SolverState, bc_type: BCType, nghost: int) -> None:
    apply_bc_func = _apply_bc_types.get(bc_type)
    if apply_bc_func is None:
        raise ValueError(f"Unsupported BC type: {bc_type}")
    return apply_bc_func(state, nghost)
