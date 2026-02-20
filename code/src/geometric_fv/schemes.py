from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from geometric_fv.slope import compute_slope
from geometric_fv.utils import simple_fixed_point


class Scheme(ABC):
    nghost: int

    @abstractmethod
    def sweep(self, u_old: np.ndarray, u_new: np.ndarray, cfl: float):
        pass


@dataclass(frozen=True)
class ImplicitUpwind(Scheme):
    nghost: int = 1

    def sweep(self, u_old: np.ndarray, u_new: np.ndarray, cfl: float):
        nghost = self.nghost
        if cfl > 0:
            for i in range(nghost, len(u_old) - nghost):
                u_new[i] = (u_old[i] + cfl * u_new[i - 1]) / (1.0 + cfl)
        else:
            for i in reversed(range(nghost, len(u_old) - nghost)):
                u_new[i] = (u_old[i] + (-cfl) * u_new[i + 1]) / (1.0 + (-cfl))


@dataclass(frozen=True)
class Box(Scheme):
    nghost: int = 1

    def sweep(self, u_old: np.ndarray, u_new: np.ndarray, cfl: float):
        nghost = self.nghost
        coeff = (1 - cfl) / (1 + cfl)
        for i in range(nghost, len(u_old) - nghost):
            u_new[i] = coeff * u_old[i] + u_old[i - 1] - coeff * u_new[i - 1]


@dataclass(frozen=True)
class SecondOrderImplicit(Scheme):
    nghost: int = 1

    def _update_cell_iter(
        self,
        u_new_i_current: float,
        u_new: np.ndarray,
        u_old: np.ndarray,
        slope: float,
        cfl: float,
        i: int,
    ) -> float:
        slope_i = compute_slope(u_old, u_new, cfl, i, u_new_i_current)

        u_new_i_next = (u_old[i] + cfl * u_new[i - 1]) / (1.0 + cfl) \
                - 0.5 * cfl * (slope_i - slope[i - 1])
        return u_new_i_next

    def sweep(self, u_old: np.ndarray, u_new: np.ndarray, cfl: float):
        nghost = self.nghost
        slope = np.zeros(len(u_old))
        coeff = (1 - cfl) / (1 + cfl)
        for i in range(nghost, len(u_old) - nghost):
            u_new_i_guess = coeff * u_old[i] + u_old[i - 1] - coeff * u_new[i - 1]

            result = simple_fixed_point(
                self._update_cell_iter,
                u_new_i_guess,
                args=(u_new, u_old, slope, cfl, i),
                tol=1e-6,
                maxiter=50,
            )
            if result.success:
                u_new[i] = result.x

                # niters = result.nit
                # print(f"Cell {i} converged in {niters} number of iterations.")
            else:
                print(f"Warning: Solver failed to converge at cell {i}.")
                print(f"Message: {result.message}")

                u_new[i] = u_new_i_guess

            slope[i] = compute_slope(u_old, u_new, cfl, i)
