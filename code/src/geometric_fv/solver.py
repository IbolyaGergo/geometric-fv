from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geometric_fv.equations import Burgers, Equation, LinearAdvection


@dataclass
class SolverState:
    u_old: np.ndarray
    u_new: np.ndarray
    slope: np.ndarray
    niter: np.ndarray
    speed: np.ndarray
    flux: np.ndarray


def _solve_linear_advection(
    eq: Equation, rhs: float, dt_dx: float, direction: str
) -> float:
    # u + dt/dx * (a * u) = rhs => u = rhs / (1 + a * dt/dx)
    return rhs / (1.0 + eq.dfdu(1) * dt_dx)


def _solve_burgers(eq: Equation, rhs: float, dt_dx: float, direction: str) -> float:
    # u + dt/dx * (u^2/2) = rhs => 0.5 * dt/dx * u^2 + u - rhs = 0
    if direction == "pos":
        return (-1.0 + np.sqrt(1.0 + 2.0 * dt_dx * rhs)) / dt_dx
    return (1.0 - np.sqrt(1 - 2 * dt_dx * rhs)) / dt_dx


_solvers = {
    LinearAdvection: _solve_linear_advection,
    Burgers: _solve_burgers,
}


def solve_for_u(eq, rhs, dt_dx, direction: str = "pos"):
    if abs(dt_dx) < 1e-14:
        return rhs
    solver = _solvers.get(type(eq)) # type: ignore
    return solver(eq, rhs, dt_dx, direction)
