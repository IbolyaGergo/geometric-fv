from abc import ABC, abstractmethod
from typing import NamedTuple

import numpy as np


# InversionResult {{{1
class InversionResult(NamedTuple):
    u: float
    is_invertible: bool


# Equation {{{1
class Equation(ABC):
    # flux {{{2
    @abstractmethod
    def flux(self, u: float) -> float:
        """Compute the physical flux f(u)."""
        pass

    # dfdu {{{2
    @abstractmethod
    def dfdu(self, u: float) -> float:
        """Compute the physical speed f'(u)."""
        pass

    # speed {{{2
    def speed(self, u1: float, u2: float) -> float:
        """Compute the numerical wave speed."""
        du = u2 - u1
        if abs(du) < 1e-14:
            return self.dfdu(u1)
        return (self.flux(u2) - self.flux(u1)) / du

    # invert_implicit() {{{2
    @abstractmethod
    def invert_implicit(
        self, rhs: float, dt_dx: float, tol: float, sweep_sign: int = 1
    ) -> InversionResult:
        """Solves u + dt/dx * f(u) = rhs. Returns (u, success)."""
        pass


# Burgers {{{1
class Burgers(Equation):
    def flux(self, u: float) -> float:
        return 0.5 * u**2

    def dfdu(self, u: float) -> float:
        return u

    def invert_implicit(
        self, rhs: float, dt_dx: float, tol: float, sweep_sign: int = 1
    ) -> InversionResult:
        if abs(dt_dx) < tol:
            return InversionResult(rhs, is_invertible=True)

        # Solve quadratic:
        # u + s * dt/dx * (u^2)/2 = rhs, where s is the sweep_sign
        discriminant = 1 + sweep_sign * 2 * dt_dx * rhs
        if discriminant > 1.0 + tol:
            u = sweep_sign * (-1.0 + np.sqrt(discriminant)) / dt_dx
            is_invertible = True
        else:
            # Fallback
            u = rhs
            is_invertible = False

        return InversionResult(u, is_invertible)


# LinearAdvection {{{1
class LinearAdvection(Equation):
    def __init__(self, a: float = 1.0):
        self.a = a

    def flux(self, u: float) -> float:
        return self.a * u

    def dfdu(self, u: float) -> float:
        return self.a

    def invert_implicit(
        self, rhs: float, dt_dx: float, tol: float, sweep_sign: int = 1
    ) -> InversionResult:
        # u + s * dt/dx * a * u = rhs, where s is the sweep_sign
        u = rhs / (1.0 + sweep_sign * self.a * dt_dx)
        return InversionResult(u, is_invertible=True)
