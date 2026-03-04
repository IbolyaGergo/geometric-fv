import numpy as np

from geometric_fv.config import ReconstConfig
from geometric_fv.enums import LimiterType, SlopeType
from geometric_fv.solver import SolverState


def _limit_slope_full(
    state: SolverState,
    i: int,
    u_new_i: float,
    slope_i: float,
) -> float:
    return 0.0


def _limit_slope_none(
    state: SolverState,
    i: int,
    u_new_i: float,
    slope_i: float,
) -> float:
    return slope_i


def _limit_slope_tvd(
    state: SolverState,
    i: int,
    u_new_i: float,
    slope_i: float,
) -> float:
    u_old = state.u_old
    u_new = state.u_new
    slope = state.slope
    cfl = state.cfl

    slope_i_1 = np.median(
        [
            slope[i - 1] - 2.0 * (u_old[i] - u_new[i - 1]) / (1.0 + cfl),
            slope[i - 1] + 2.0 * (u_old[i] - u_new[i - 1]) / ((cfl) * (1.0 + cfl)),
            slope_i,
        ]
    )

    slope_i_lim = np.median(
        [0.0, slope_i_1, 2.0 * (u_old[i + 1] - u_new_i) / (1.0 + cfl)]
    )

    return slope_i_lim


def _limit_slope_tvd_suff(
    state: SolverState,
    i: int,
    u_new_i: float,
    slope_i: float,
) -> float:
    u_old = state.u_old
    u_new = state.u_new
    slope = state.slope
    cfl = state.cfl

    slope_i_1 = np.median([0.0, 2.0 * (u_old[i + 1] - u_new_i) / (1.0 + cfl), slope_i])

    slope_i_lim = np.median(
        [0.0, (2.0 / cfl) * (u_old[i] - u_new[i - 1]) / (1.0 + cfl), slope_i_1]
    )

    return slope_i_lim


def _compute_slope_box(state: SolverState, i: int, u_new_i: float) -> float:
    u_old = state.u_old
    cfl = state.cfl

    slope_i = (u_old[i] - u_new_i) / cfl

    return slope_i


_limit_slope_types = {
    LimiterType.FULL: _limit_slope_full,
    LimiterType.NONE: _limit_slope_none,
    LimiterType.TVD: _limit_slope_tvd,
    LimiterType.TVD_SUFF: _limit_slope_tvd_suff,
}

_compute_slope_types = {
    SlopeType.BOX: _compute_slope_box,
}


def compute_slope(
    state: SolverState, i: int, u_new_i: float, reconst_config: ReconstConfig
) -> float:
    slope_type = reconst_config.slope_type
    compute_slope_func = _compute_slope_types.get(slope_type)
    if compute_slope_func is None:
        raise ValueError(f"Unsupported slope type: {slope_type}")

    slope_i = compute_slope_func(state, i, u_new_i)

    limiter_type = reconst_config.limiter_type
    limit_slope_func = _limit_slope_types.get(limiter_type)
    if limit_slope_func is None:
        raise ValueError(f"Unsupported limiter type: {limiter_type}")

    slope_i_lim = limit_slope_func(state, i, u_new_i, slope_i)

    return slope_i_lim
