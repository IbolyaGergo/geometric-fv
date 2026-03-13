import numpy as np
from typing import Callable

from geometric_fv.config import ReconstConfig
from geometric_fv.enums import GuessType, LimiterType, SlopeType, FluxType
from geometric_fv.solver import SolverState


# LIMITER {{{1
# _limit_slope_full() {{{2
def _limit_slope_full(
    state: SolverState,
    i: int,
    u_new_i: float,
    slope_i: float,
) -> float:
    return 0.0


# _limit_slope_none() {{{2
def _limit_slope_none(
    state: SolverState,
    i: int,
    u_new_i: float,
    slope_i: float,
) -> float:
    return slope_i


# _limit_slope_tvd() {{{2
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

    i_upw = i - 1 if cfl > 0 else i + 1
    i_dwn = i + 1 if cfl > 0 else i - 1

    slope_i_1 = np.median(
        [
            slope[i_upw]
            - np.sign(cfl) * 2.0 * (u_old[i] - u_new[i_upw]) / (1.0 + abs(cfl)),
            slope[i_upw] + 2.0 * (u_old[i] - u_new[i_upw]) / (cfl * (1.0 + abs(cfl))),
            slope_i,
        ]
    )

    slope_i_lim = np.median(
        [
            0.0,
            slope_i_1,
            np.sign(cfl) * 2.0 * (u_old[i_dwn] - u_new_i) / (1.0 + abs(cfl)),
        ]
    )

    return slope_i_lim


# _limit_slope_tvd_suff() {{{2
def _limit_slope_tvd_suff(
    state: SolverState,
    i: int,
    u_new_i: float,
    slope_i: float,
) -> float:
    u_old = state.u_old
    u_new = state.u_new
    cfl = state.cfl

    i_upw = i - 1 if cfl > 0 else i + 1
    i_dwn = i + 1 if cfl > 0 else i - 1

    slope_i_1 = np.median(
        [
            0.0,
            np.sign(cfl) * 2.0 * (u_old[i_dwn] - u_new_i) / (1.0 + abs(cfl)),
            (2.0 / cfl) * (u_old[i] - u_new[i_upw]) / (1.0 + abs(cfl)),
        ]
    )

    slope_i_lim = np.median([0.0, slope_i, slope_i_1])

    return slope_i_lim


# _limit_slope_types {{{2
_limit_slope_types = {
    LimiterType.FULL: _limit_slope_full,
    LimiterType.NONE: _limit_slope_none,
    LimiterType.TVD: _limit_slope_tvd,
    LimiterType.TVD_SUFF: _limit_slope_tvd_suff,
}


# SLOPE {{{1
# _compute_slope_box() {{{2
def _compute_slope_box(state: SolverState, i: int, u_new_i: float) -> float:
    u_old = state.u_old
    cfl = state.cfl

    if np.not_equal(cfl, 0.0):
        slope_i = (u_old[i] - u_new_i) / cfl
    else:
        slope_i = 0.0

    return slope_i


# _compute_slope_types() {{{2
_compute_slope_types = {
    SlopeType.BOX: _compute_slope_box,
}


# compute_slope() {{{2
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


# GUESS {{{1
# _compute_guess_box() {{{2
def _compute_guess_box(state: SolverState, i: int) -> float:
    u_old = state.u_old
    u_new = state.u_new
    cfl = state.cfl

    coeff = (1 - abs(cfl)) / (1 + abs(cfl))
    i_upw = i - 1 if cfl > 0 else i + 1
    u_new_i_guess = coeff * u_old[i] + u_old[i_upw] - coeff * u_new[i_upw]

    return u_new_i_guess


# _compute_guess_implicit_upwind() {{{2
def _compute_guess_implicit_upwind(state: SolverState, i: int) -> float:
    u_old = state.u_old
    u_new = state.u_new
    cfl = state.cfl

    i_upw = i - 1 if state.cfl > 0 else i + 1
    u_new_i_guess = (u_old[i] + abs(cfl) * u_new[i_upw]) / (1.0 + abs(cfl))

    return u_new_i_guess


# _compute_guess_types {{{2
_compute_guess_types = {
    GuessType.IMPLICIT_UPWIND: _compute_guess_implicit_upwind,
    GuessType.BOX: _compute_guess_box,
}


# compute_guess() {{{2
def compute_guess(state: SolverState, i: int, reconst_config: ReconstConfig) -> float:
    guess_type = reconst_config.guess_type
    compute_guess_func = _compute_guess_types.get(guess_type)
    if compute_guess_func is None:
        raise ValueError(f"Unsupported guess type: {guess_type}")

    u_guess = compute_guess_func(state, i)

    if reconst_config.limiter_type != LimiterType.NONE:
        i_upw = i - 1 if state.cfl > 0 else i + 1
        u_guess = np.median([u_guess, state.u_old[i], state.u_new[i_upw]])

    return u_guess

# FLUX {{{1
def _flux_burgers(u: float) -> float:
    return 0.5 * u * u
def _dfdu_burgers(u: float) -> float:
    return u
# def _flux_linear(u: float) -> float:
#     return u
# def _dfdu_linear(u: float) -> float:
#     return 1

_flux_types = {
        FluxType.BURGERS: _flux_burgers,
        # FluxType.LINEAR_ADVECTION: _flux_linear,
        }

def compute_flux(u: float, flux_type: FluxType) -> float:
    flux_func = _flux_types.get(flux_type)
    if flux_func is None:
        raise ValueError(f"Unsupported flux type: {flux_type}")
    return flux_func(u)

# SPEED {{{1
# _compute_speed_box() {{{2
def _compute_speed_box(state: SolverState, i: int, u_new_i: float, compute_flux: Callable[[float], float]):
    u_old = state.u_old
    u_new = state.u_new
    cfl = state.cfl

    dfdu_exact = _dfdu_burgers

    df = compute_flux(u_old[i]) - compute_flux(u_new_i)
    du = u_old[i] - u_new_i
    if np.equal(du, 0.0):
        return dfdu_exact(u_old[i]) # TODO: only works for Burgers'
    else:
        return df / du

# _compute_speed_types {{{2
_compute_speed_types = {
    SlopeType.BOX: _compute_speed_box,
}


# compute_speed() {{{1
def compute_speed(state: SolverState, i: int, u_new_i: float,
                  reconst_config: ReconstConfig) -> float:
    slope_type = reconst_config.slope_type
    compute_speed_func = _compute_speed_types.get(slope_type)
    if compute_speed_func is None:
        raise ValueError(f"Unsupported slope type: {slope_type}")
    flux = _flux_burgers
    return compute_speed_func(state, i, u_new_i, flux)
