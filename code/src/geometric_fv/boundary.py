from enum import Enum

import numpy as np


class BCType(Enum):
    CONSTANT_EXTEND = "constant_extend"
    QUASI_PERIODIC = "quasi_periodic"


def _apply_bc_constant_extend(
    u_old: np.ndarray, u_new: np.ndarray, nghost: int, clf: float
):
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
def _apply_bc_quasi_periodic(
    u_old: np.ndarray, u_new: np.ndarray, nghost: int, cfl: float
):
    for i in range(nghost):
        u_old[i] = u_old[-2 * nghost + i]
        u_old[-1 - i] = u_old[2 * nghost - 1 - i]

    if cfl > 0.0:
        u_new[nghost - 1] =\
            (1 + int(np.floor(cfl)) - cfl) * u_old[-2 - int(np.floor(cfl))] +\
                (cfl - int(np.floor(cfl))) * u_old[-3 - int(np.floor(cfl))]
    else:
        u_new[-1] =\
            (1 + int(np.floor(-cfl)) - (-cfl)) * u_old[1 + int(np.floor(-cfl))] +\
                  (-cfl - int(np.floor(-cfl))) * u_old[2 + int(np.floor(-cfl))]


_apply_bc_types = {
    BCType.CONSTANT_EXTEND: _apply_bc_constant_extend,
    BCType.QUASI_PERIODIC: _apply_bc_quasi_periodic,
}


def apply_bc(
    bc_type: BCType, u_old: np.ndarray, u_new: np.ndarray, nghost: int, cfl: float
):
    apply_bc_func = _apply_bc_types.get(bc_type)
    if apply_bc_func is None:
        raise ValueError(f"Unsupported BC type: {bc_type}")
    return apply_bc_func(u_old, u_new, nghost, cfl)
