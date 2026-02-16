from enum import Enum

import numpy as np


class SchemeType(Enum):
    IMPLICIT_UPWIND = "implicit_upwind"


def _update_cell_implicit_upwind(
    idx: int, u_old: np.ndarray, u_new: np.ndarray, cfl: float, get_bc_func: callable
) -> float:
    if idx == 0:
        u_new_BC_left = get_bc_func(u_old, cfl)[0]
        return (u_old[0] + cfl * u_new_BC_left) / (1 + cfl)

    return (u_old[idx] + cfl * u_new[idx - 1]) / (1 + cfl)


_update_cell_schemes = {SchemeType.IMPLICIT_UPWIND: _update_cell_implicit_upwind}


def update_cell(
    idx: int,
    u_old: np.ndarray,
    u_new: np.ndarray,
    cfl: float,
    scheme_type: SchemeType = SchemeType.IMPLICIT_UPWIND,
    get_bc_func: callable = None,
) -> float:
    """
    Solves for u_i^{n+1} using the specified scheme.

    Args:
        cfl: Courant number
        scheme_type: Numerical scheme.

    Returns:
        u_i^{n+1}
    """
    update = _update_cell_schemes.get(scheme_type)
    if update is None:
        raise ValueError(f"Unsupported scheme type: {scheme_type}")
    return update(idx, u_old, u_new, cfl, get_bc_func)


def sweep(
    ncells: int,
    u_old: np.ndarray,
    cfl: float,
    scheme_type: SchemeType = SchemeType.IMPLICIT_UPWIND,
    get_bc_func: callable = None,
) -> np.ndarray:
    u_new = np.zeros(ncells)

    if get_bc_func == None:
        get_bc_func = lambda u_old, cfl: np.array([0.0, 0.0])

    for i in range(ncells):
        u_new[i] = update_cell(i, u_old, u_new, cfl, scheme_type, get_bc_func)

    return u_new
