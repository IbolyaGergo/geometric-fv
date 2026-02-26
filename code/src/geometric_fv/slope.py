from enum import Enum

import numpy as np


class SlopeType(Enum):
    BOX = 0


class LimiterType(Enum):
    FULL = 0
    NONE = 1
    TVD = 2


def _limit_slope_full(
    state,
    i: int,
    u_new_i: float,
    slope_i_current: float,
) -> float:
    return 0.0


def _limit_slope_none(
    state,
    i: int,
    u_new_i: float,
    slope_i_current: float,
) -> float:
    return slope_i_current


def _limit_slope_tvd(
    state,
    i: int,
    u_new_i: float,
    slope_i_current: float,
) -> float:
    u_old_i = state.u_old[i]
    u_old_ip1 = state.u_old[i + 1]

    u_new_im1 = state.u_new[i - 1]

    slope_i = slope_i_current
    slope_im1 = state.slope[i - 1]
    cfl = state.cfl

    slope_i_1 = np.median(
        [
            slope_im1 - 2.0 * (u_old_i - u_new_im1) / (1.0 + cfl),
            slope_im1 + 2.0 * (u_old_i - u_new_im1) / ((cfl) * (1.0 + cfl)),
            slope_i,
        ]
    )

    slope_i_lim = np.median([0.0, slope_i_1, 2.0 * (u_old_ip1 - u_new_i) / (1.0 + cfl)])

    return slope_i_lim


def _compute_slope_box(state, i: int, u_new_i: float) -> float:
    u_old_i = state.u_old[i]
    cfl = state.cfl

    slope_i = (u_old_i - u_new_i) / cfl

    return slope_i

_limit_slope_types = {
    LimiterType.FULL: _limit_slope_full,
    LimiterType.NONE: _limit_slope_none,
    LimiterType.TVD: _limit_slope_tvd,
}

_compute_slope_types = {
    SlopeType.BOX: _compute_slope_box,
}


def compute_slope(
    state,
    i: int,
    u_new_i: float,
    slope_type: SlopeType = SlopeType.BOX,
    limiter_type=LimiterType.NONE,
) -> float:
    compute_slope_func = _compute_slope_types.get(slope_type)
    if compute_slope_func is None:
        raise ValueError(f"Unsupported slope type: {slope_type}")

    slope_i = compute_slope_func(state, i, u_new_i)

    limit_slope_func = _limit_slope_types.get(limiter_type)
    if limit_slope_func is None:
        raise ValueError(f"Unsupported limiter type: {limiter_type}")

    slope_i_lim = limit_slope_func(state, i, u_new_i, slope_i)

    return slope_i_lim
