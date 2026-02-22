from enum import Enum

import numpy as np


class SlopeType(Enum):
    ZERO = 0
    BOX = 1
    TVD_BOX = 2


def _compute_slope_limiter_full(
        u_old: np.ndarray, u_new: np.ndarray, slope: np.ndarray, cfl: float, i: int, u_new_i_current: float
) -> float:
    return 0.0


def _compute_slope_limiter_box(
        u_old: np.ndarray, u_new: np.ndarray, slope: np.ndarray, cfl: float, i: int, u_new_i_current: float
) -> float:
    u_new_i = u_new_i_current if u_new_i_current is not np.nan else u_new[i]
    slope_i = (u_old[i] - u_new_i) / cfl

    return slope_i

def _compute_slope_limiter_tvd_box(
        u_old: np.ndarray, u_new: np.ndarray, slope: np.ndarray, cfl: float, i: int, u_new_i_current: float
) -> float:
    u_new_i = u_new_i_current if u_new_i_current is not np.nan else u_new[i]

    slope_i = (u_old[i] - u_new_i) / cfl
    slope_im1 = slope[i-1]

    slope_i_1 = np.median([
        slope_im1 - 2.0 * (u_old[i] - u_new[i-1]) / (1.0 + cfl),
        slope_im1 + 2.0 * (u_old[i] - u_new[i-1]) / ((cfl) * (1.0 + cfl)),
        slope_i
        ])
    slope_i_lim = np.median([
        0.0,
        slope_i_1,
        2.0 * (u_old[i+1] - u_new_i) / (1.0 + cfl)
        ])

    print(f"tvd slope called")

    return slope_i_lim

_compute_slope_types = {
    SlopeType.ZERO: _compute_slope_limiter_full,
    SlopeType.BOX: _compute_slope_limiter_box,
    SlopeType.TVD_BOX: _compute_slope_limiter_tvd_box,
}


def compute_slope(
    u_old: np.ndarray,
    u_new: np.ndarray,
    slope: np.ndarray,
    cfl: float,
    i: int,
    u_new_i_current: float = np.nan,
    slope_type: SlopeType = SlopeType.TVD_BOX
) -> float:
    compute_slope_func = _compute_slope_types.get(slope_type)
    if compute_slope_func is None:
        raise ValueError(f"Unsupported slope type: {slope_type}")
    return compute_slope_func(u_old, u_new, slope, cfl, i, u_new_i_current)
