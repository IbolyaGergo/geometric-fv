"""Benchmark problems"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import fsolve

from geometric_fv.equations import Burgers, Equation


# Problem {{{1
@dataclass
class Problem(ABC):
    """Base class for benchmark problems.

    A Problem bundles an equation with its initial condition and provides the
    exact analytical solution at a given time t.
    """

    equation: Equation
    x_min: float = 0.0
    x_max: float = 1.0

    @abstractmethod
    def u0(self, x: np.ndarray) -> np.ndarray:
        """Exact solution at t=0."""
        pass

    @abstractmethod
    def exact(self, x: np.ndarray, t: float) -> np.ndarray:
        """Exact solution at time t."""
        pass


# BurgersSmooth {{{1
@dataclass
class BurgersSmooth(Problem):
    """Smooth solution to the Burgers equation.

    Uses the method of characteristics to solve u(x, t) = u0(x - u*t)
    implicitly. This solution is valid only before shock formation.
    """

    equation: Equation = Burgers()
    u0_func: Callable[[np.ndarray], np.ndarray] = lambda x: (
        0.5 + 0.5 * np.exp(-100 * (x - 0.5) ** 2)
    )

    @property
    def t_shock(self) -> float:
        """Estimate the time of shock formation: t_shock = -1/min(u0')."""
        x = np.linspace(self.x_min, self.x_max, 2000)
        u0 = self.u0_func(x)
        dudx = np.gradient(u0, x)
        min_deriv = np.min(dudx)
        if min_deriv >= 0:
            return np.inf
        return -1.0 / min_deriv

    def u0(self, x: np.ndarray) -> np.ndarray:
        return self.u0_func(x)

    def exact(self, x: np.ndarray, t: float) -> np.ndarray:
        """Computes the exact solution using the method of characteristics.

        Solves the implicit equation u = u0(x - u*t).
        """
        if t > self.t_shock + 1e-12:
            raise ValueError(
                f"Time t={t:.4f} is greater than shock formation time "
                f"t_shock={self.t_shock:.4f}. Solution is no longer smooth."
            )

        if t < 1e-14:
            return self.u0(x)

        def residual(u, xi):
            return u - self.u0_func(xi - u * t)

        u_exact = np.zeros_like(x)
        # Use u0(x) as the initial guess
        for i, xi in enumerate(x):
            sol = fsolve(residual, x0=self.u0_func(xi), args=(xi,))
            u_exact[i] = sol[0]

        return u_exact
