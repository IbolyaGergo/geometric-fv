from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np

from geometric_fv.config import SolverConfig
from geometric_fv.enums import LimiterType
from geometric_fv.slope import compute_slope
from geometric_fv.solver import SolverState
from geometric_fv.utils import simple_fixed_point


class Scheme(ABC):
    nghost: int
    config: SolverConfig

    def allocate_state(self, u0: np.ndarray, cfl: float) -> SolverState:
        """Creates a SolverState from an existing array."""
        u_padded = np.pad(u0, (self.nghost, self.nghost), mode="constant")
        return SolverState(
            u_old=u_padded.copy(),
            u_new=u_padded.copy(),
            slope=np.zeros_like(u_padded),
            niter=np.zeros_like(u_padded, dtype=int),
            cfl=cfl,
        )

    def init_state(
        self, func: Callable[[np.ndarray], np.ndarray], cfl: float
    ) -> SolverState:
        """
        Generates u0 using func and the mesh defined in the scheme's config.
        """
        # To avoid circular dependency
        from geometric_fv.mesh import Mesh1D

        mesh = Mesh1D.uniform(self.config.mesh)

        u0 = func(mesh.centers)

        return self.allocate_state(u0, cfl)

    def apply_bc(self, state: SolverState) -> None:
        # Local import to avoid circular dependency at the top of the file
        from geometric_fv.boundary import apply_bc as _apply_bc_kernel

        _apply_bc_kernel(
            state=state,
            nghost=self.nghost,
            config=self.config.boundary,
            reconst_config=self.config.reconst,
        )

    @abstractmethod
    def sweep(self, state: SolverState):
        pass


@dataclass(frozen=True)
class SecondOrderImplicit(Scheme):
    nghost: int = 2
    config: SolverConfig = SolverConfig()

    def _update_cell_guess(self, state: SolverState, i: int) -> float:
        u_old = state.u_old
        u_new = state.u_new
        cfl = state.cfl

        coeff = (1 - cfl) / (1 + cfl)
        u_new_i_guess = coeff * u_old[i] + u_old[i - 1] - coeff * u_new[i - 1]

        limiter_type = self.config.reconst.limiter_type
        if limiter_type is not LimiterType.NONE:
            u_new_i_guess = np.median([u_new_i_guess, u_old[i], u_new[i - 1]])

        return u_new_i_guess

    def _update_cell_iter(
        self,
        u_new_i_current: float,
        state: SolverState,
        i: int,
    ) -> float:
        u_old = state.u_old
        u_new = state.u_new
        slope = state.slope

        cfl = state.cfl

        slope[i] = compute_slope(
            state, i=i, u_new_i=u_new_i_current, reconst_config=self.config.reconst
        )

        # fmt: off
        u_new_i_next = (u_old[i] + cfl * u_new[i - 1]) / (1.0 + cfl) \
                - 0.5 * cfl * (slope[i] - slope[i - 1])
        # fmt: on
        return u_new_i_next

    def sweep(self, state: SolverState):
        nghost = self.nghost
        if state.niter is None:
            state.niter = np.zeros_like(state.u_old, dtype=int)

        for i in range(nghost, len(state.u_old) - nghost):
            u_new_i_guess = self._update_cell_guess(state, i=i)

            result = simple_fixed_point(
                self._update_cell_iter,
                u_new_i_guess,
                args=(state, i),
                tol=self.config.iteration.tol,
                maxiter=self.config.iteration.maxiter,
            )
            if result.success:
                state.u_new[i] = result.x
                state.niter[i] = result.nit

                # print(f"Cell {i} converged in {state.niter[i]} number of iterations.")
            else:
                print(f"Warning: Solver failed to converge at cell {i}.")
                print(f"Message: {result.message}")

                state.u_new[i] = u_new_i_guess
