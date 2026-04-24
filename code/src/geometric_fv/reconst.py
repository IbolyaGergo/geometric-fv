import numpy as np

from geometric_fv.config import SolverConfig
from geometric_fv.enums import (
    AvgSpeedType,
    FluxCorrType,
    GuessType,
    LimiterType,
    SlopeType,
)
from geometric_fv.equations import Equation
from geometric_fv.solver import SolverState


# LIMITER {{{1
# _limit_slope_full() {{{2
def _limit_slope_full(
    state: SolverState,
    i: int,
    u_new_i: float,
    slope_i: float,
    dt_dx: float,
    eq: Equation,
) -> float:
    return 0.0


# _limit_slope_none() {{{2
def _limit_slope_none(
    state: SolverState,
    i: int,
    u_new_i: float,
    slope_i: float,
    dt_dx: float,
    eq: Equation,
) -> float:
    return slope_i


# _limit_slope_tvd() {{{2
def _limit_slope_tvd(
    state: SolverState,
    i: int,
    u_new_i: float,
    slope_i: float,
    dt_dx: float,
    eq: Equation,
) -> float:
    u_old = state.u_old
    u_new = state.u_new
    slope = state.slope

    cfl = eq.dfdu(u_new_i) * dt_dx
    i_upw = i - 1 if cfl > 0 else i + 1
    i_dwn = i + 1 if cfl > 0 else i - 1

    slope_i_1 = np.median(
        [
            slope[i_upw]
            - np.sign(cfl) * 2.0 * (u_old[i] - u_new[i_upw]) / (1.0 + abs(cfl)),
            slope[i_upw]
            + 2.0 * (u_old[i] - u_new[i_upw]) / (cfl * (1.0 + abs(cfl))),
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
    dt_dx: float,
    eq: Equation,
) -> float:
    u_old = state.u_old
    u_new = state.u_new

    cfl = eq.dfdu(u_new_i) * dt_dx
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
def _compute_slope_box(
    state: SolverState, i: int, u_new_i: float, dt_dx: float, eq: Equation
) -> float:
    u_old = state.u_old

    cfl = eq.dfdu(u_new_i) * dt_dx
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
    state: SolverState, i: int, u_new_i: float, config: SolverConfig
) -> float:
    slope_type = config.reconst.slope_type
    compute_slope_func = _compute_slope_types.get(slope_type)
    if compute_slope_func is None:
        raise ValueError(f"Unsupported slope type: {slope_type}")

    dt_dx = config.dt_dx
    eq = config.equation
    slope_i = compute_slope_func(state, i, u_new_i, dt_dx, eq)

    limiter_type = config.reconst.limiter_type
    limit_slope_func = _limit_slope_types.get(limiter_type)
    if limit_slope_func is None:
        raise ValueError(f"Unsupported limiter type: {limiter_type}")

    slope_i_lim = limit_slope_func(state, i, u_new_i, slope_i, dt_dx, eq)

    return slope_i_lim


# AVG SPEED {{{1
# _compute_avg_speed_impl_upwind() {{{2
def _compute_avg_speed_impl_upwind(
    state: SolverState, i: int, u_new_i: float, eq: Equation
):
    return eq.dfdu(u_new_i)


# _avg_speed_types {{{2
_avg_speed_types = {
    AvgSpeedType.IMPLICIT_UPWIND: _compute_avg_speed_impl_upwind,
}


# FLUX LIMITER {{{1
# _limit_flux_corr_full() {{{2
def _limit_flux_corr_full(
    state: SolverState,
    i: int,
    u_new_i: float,
    flux_corr_i: float,
    dt_dx: float,
    eq: Equation,
) -> float:
    return 0.0


# _limit_flux_corr_none() {{{2
def _limit_flux_corr_none(
    state: SolverState,
    i: int,
    u_new_i: float,
    flux_corr_i: float,
    dt_dx: float,
    eq: Equation,
) -> float:
    return flux_corr_i


# _limit_flux_corr_tvd() {{{2
def _limit_flux_corr_tvd(
    state: SolverState,
    i: int,
    u_new_i: float,
    flux_corr_i: float,
    dt_dx: float,
    eq: Equation,
) -> float:
    u_new = state.u_new
    u_old = state.u_old

    flux_in = state.flux[i - 1]

    flux_corr_i_1 = np.median(
        [
            flux_corr_i,
            flux_in - eq.flux(u_new_i) - (u_new[i - 1] - u_old[i]) / dt_dx,
            flux_in - eq.flux(u_new_i),
        ]
    )

    flux_corr_i_lim = np.median(
        [
            flux_corr_i_1,
            0.0,
            eq.flux(u_old[i + 1]) - eq.flux(u_new_i),
        ]
    )
    return flux_corr_i_lim


# _limit_flux_corr_types {{{2
_limit_flux_corr_types = {
    LimiterType.FULL: _limit_flux_corr_full,
    LimiterType.NONE: _limit_flux_corr_none,
    LimiterType.TVD: _limit_flux_corr_tvd,
}


# FLUX {{{1
# _compute_flux_corr_geom() {{{2
def _compute_flux_corr_geom(
    state: SolverState, i: int, u_new_i: float, config: SolverConfig
) -> float:
    dt_dx = config.dt_dx
    eq = config.equation

    # Slope
    slope_type = config.reconst.slope_type
    compute_slope_func = _compute_slope_types.get(slope_type)
    if compute_slope_func is None:
        raise ValueError(f"Unsupported slope type: {slope_type}")
    slope_i = compute_slope_func(state, i, u_new_i, dt_dx, eq)

    # Speed
    # Avg
    avg_speed_type = config.reconst.avg_speed_type
    compute_avg_speed_func = _avg_speed_types.get(avg_speed_type)
    if compute_avg_speed_func is None:
        raise ValueError(f"Unsupported avg speed type: {avg_speed_type}")
    avg_speed_i = compute_avg_speed_func(state, i, u_new_i, eq)

    # Char
    char_speed_i = eq.dfdu(u_new_i)

    # Flux correction
    # Unlimited
    flux_corr = (
        avg_speed_i * slope_i * (1 + (2 * char_speed_i - avg_speed_i) * dt_dx) * 0.5
    )

    return flux_corr


# _flux_corr_types {{{2
_flux_corr_types = {
    FluxCorrType.GEOMETRIC: _compute_flux_corr_geom,
}


# compute_flux_corr() {{{2
def compute_flux_corr(state: SolverState, i: int, u_new_i: float, config: SolverConfig):
    flux_corr_type = config.reconst.flux_corr_type
    compute_flux_corr_func = _flux_corr_types.get(flux_corr_type)
    if compute_flux_corr_func is None:
        raise ValueError(f"Unsupported flux correction type: {flux_corr_type}")

    # Raw flux correction
    flux_corr = compute_flux_corr_func(state, i, u_new_i, config)

    # Limit
    limiter_type = config.reconst.limiter_type
    limit_flux_corr_func = _limit_flux_corr_types.get(limiter_type)
    if limit_flux_corr_func is None:
        raise ValueError(f"Unsupported limiter type: {limiter_type}")

    dt_dx = config.dt_dx
    eq = config.equation
    return limit_flux_corr_func(state, i, u_new_i, flux_corr, dt_dx, eq)


# GUESS {{{1
# _compute_guess_box() {{{2
def _compute_guess_box(state: SolverState, i: int, dt_dx: float, eq: Equation) -> float:
    u_old = state.u_old
    u_new = state.u_new

    cfl = eq.dfdu(u_old[i]) * dt_dx
    coeff = (1 - abs(cfl)) / (1 + abs(cfl))
    i_upw = i - 1 if cfl > 0 else i + 1
    u_new_i_guess = coeff * u_old[i] + u_old[i_upw] - coeff * u_new[i_upw]

    return u_new_i_guess


# _compute_guess_implicit_upwind() {{{2
def _compute_guess_implicit_upwind(state: SolverState, i: int, dt_dx: float,
                                   eq: Equation) -> float:
    u_old = state.u_old
    u_new = state.u_new

    cfl = eq.dfdu(u_old[i]) * dt_dx
    i_upw = i - 1 if cfl > 0 else i + 1
    u_new_i_guess = (u_old[i] + abs(cfl) * u_new[i_upw]) / (1.0 + abs(cfl))

    return u_new_i_guess


# _compute_guess_types {{{2
_compute_guess_types = {
    GuessType.IMPLICIT_UPWIND: _compute_guess_implicit_upwind,
    GuessType.BOX: _compute_guess_box,
}


# compute_guess() {{{2
def compute_guess(state: SolverState, i: int, config: SolverConfig) -> float:
    guess_type = config.reconst.guess_type
    compute_guess_func = _compute_guess_types.get(guess_type)
    if compute_guess_func is None:
        raise ValueError(f"Unsupported guess type: {guess_type}")

    dt_dx = config.dt_dx
    eq = config.equation
    u_guess = compute_guess_func(state, i, dt_dx, eq)

    cfl = eq.dfdu(u_guess) * dt_dx
    if config.reconst.limiter_type != LimiterType.NONE:
        i_upw = i - 1 if cfl > 0 else i + 1
        u_guess = np.median([u_guess, state.u_old[i], state.u_new[i_upw]])

    return u_guess
