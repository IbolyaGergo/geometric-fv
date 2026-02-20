import numpy as np


def compute_slope(
    u_old: np.ndarray,
    u_new: np.ndarray,
    cfl: float,
    i: int,
    u_new_i_current: float = None,
):
    u_new_i = u_new_i_current if u_new_i_current is not None else u_new[i]
    slope_i = (u_old[i] - u_new_i) / cfl
    return slope_i
