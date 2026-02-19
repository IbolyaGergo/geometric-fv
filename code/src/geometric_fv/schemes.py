from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
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

    def func(
        self,
        u_new_i_guess: float,
        u_old_i: float,
        u_new_guess_im1: float,
        grad_im1: float, cfl: float
    ) -> float:
        grad_i = (u_old_i - u_new_i_guess) / cfl
        u_new_i_guess = (u_old_i + cfl * u_new_guess_im1) / (1.0 + cfl) \
                -0.5 * cfl * (grad_i - grad_im1)
        return u_new_i_guess

    def sweep(self, u_old: np.ndarray, u_new: np.ndarray, cfl: float):
        nghost = self.nghost
        grad = np.zeros(len(u_old))
        coeff = (1 - cfl) / (1 + cfl)
        for i in range(nghost, len(u_old) - nghost):
            u_new_i_guess = coeff * u_old[i] + u_old[i - 1] - coeff * u_new[i - 1]

            result = simple_fixed_point(
                    self.func,
                    u_new_i_guess,
                    args=(u_old[i], u_new[i - 1], grad[i - 1], cfl),
                    tol=1e-6,
                    maxiter=50
                    )
            if result.success:
                u_new[i] = result.x
                niters = result.nit

                print(f"Cell {i} converged in {niters} number of iterations.")
            else:
                print(f"Warning: Solver failed to converge at cell {i}.")
                print(f"Message: {result.message}")

                u_new[i] = u_new_i_guess

            grad[i] = (u_old[i] - u_new[i]) / cfl
