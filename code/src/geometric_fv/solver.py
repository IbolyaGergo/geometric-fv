from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geometric_fv.equations import Burgers, Equation, LinearAdvection


@dataclass
class SolverState:
    u_old: np.ndarray
    u_new: np.ndarray
    slope: np.ndarray
    niter: np.ndarray | None = None
    speed: np.ndarray | None = None


def _solve_linear_advection(eq: Equation, rhs: float, dt_dx: float) -> float:
    # u + dt/dx * (a * u) = rhs => u = rhs / (1 + a * dt/dx)
    return rhs / (1.0 + eq.a * dt_dx)

def _solve_burgers(eq: Equation, rhs: float, dt_dx: float) -> float:
    # u + dt/dx * (u^2/2) = rhs => 0.5 * dt/dx * u^2 + u - rhs = 0
    return (-1.0 + np.sqrt(1.0 + 2.0 * dt_dx * rhs)) / dt_dx

_solvers = {
        LinearAdvection: _solve_linear_advection,
        Burgers: _solve_burgers,
        }

def solve_for_u(eq, rhs, dt_dx):
    if abs(dt_dx) < 1e-14:
        return rhs
    solver = _solvers.get(type(eq))
    return solver(eq, rhs, dt_dx)
