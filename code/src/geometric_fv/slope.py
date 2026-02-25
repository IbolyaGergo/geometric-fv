from enum import Enum

import numpy as np


class SlopeType(Enum):
    ZERO = 0
    BOX = 1
    TVD_BOX = 2


def _compute_slope_limiter_full(
        state,
    i: int,
    u_new_i_current: float,
) -> float:
    return 0.0


def _compute_slope_limiter_box(
        state,
    i: int,
    u_new_i_current: float,
) -> float:
    u_new_i = u_new_i_current
    u_old_i = state.u_old[i]
    cfl = state.cfl

    slope_i = (u_old_i - u_new_i) / cfl

    return slope_i


def _compute_slope_limiter_tvd_box(
    state,
    i: int,
    u_new_i_current: float,
) -> float:
    u_new_i = u_new_i_current
    u_old_i = state.u_old[i]
    cfl = state.cfl

    slope_i_current = (u_old_i - u_new_i) / cfl

    slope_i_lim = _limit_slope_tvd(
        state,
        i=i,
        u_new_i_current=u_new_i_current,
        slope_i_current=slope_i_current,
    )

    return slope_i_lim


_compute_slope_types = {
    SlopeType.ZERO: _compute_slope_limiter_full,
    SlopeType.BOX: _compute_slope_limiter_box,
    SlopeType.TVD_BOX: _compute_slope_limiter_tvd_box,
}

_limit_slope_types = {
        LimiterType.TVD: _limit_slope_tvd,
        }


def compute_slope(
    state,
    i: int,
    u_new_i_current: float,
    slope_type: SlopeType = SlopeType.TVD_BOX,
) -> float:
    compute_slope_func = _compute_slope_types.get(slope_type)
    if compute_slope_func is None:
        raise ValueError(f"Unsupported slope type: {slope_type}")
    return compute_slope_func(state, i, u_new_i_current)
