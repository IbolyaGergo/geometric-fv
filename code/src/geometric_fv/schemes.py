from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np

from geometric_fv.config import SolverConfig
from geometric_fv.slope import compute_guess, compute_slope
from geometric_fv.solver import SolverState, solve_for_u
from geometric_fv.utils import simple_fixed_point


# Scheme() {{{1
class Scheme(ABC):
    nghost: int
    config: SolverConfig

    # allocate_state() {{{2
    def allocate_state(self, u0: np.ndarray) -> SolverState:
        """Creates a SolverState from an existing array."""
        u_padded = np.pad(u0, (self.nghost, self.nghost), mode="constant")
        return SolverState(
            u_old=u_padded.copy(),
            u_new=u_padded.copy(),
            slope=np.zeros_like(u_padded),
            speed=np.zeros_like(u_padded),
            niter=np.zeros_like(u_padded, dtype=int),
        )

    # init_state() {{{2
    def init_state(self, func: Callable[[np.ndarray], np.ndarray]) -> SolverState:
        """
        Generates u0 using func and the mesh defined in the scheme's config.
        """
        mesh = self.config.mesh.create_mesh()

        u0 = func(mesh.centers)

        return self.allocate_state(u0)

    # apply_bc() {{{2
    def apply_bc(self, state: SolverState) -> None:
        # Local import to avoid circular dependency at the top of the file
        from geometric_fv.boundary import apply_bc as _apply_bc_kernel

        _apply_bc_kernel(
            state=state,
            nghost=self.nghost,
            config=self.config,
        )

    # cell_indices() {{{2
    def cell_indices(self, state: SolverState) -> range | reversed[int]:
        """
        Returns an iterator over the internal cell indices.

        Args:
            state: The current state to get the total length.
            reverse: If True, sweeps from right-to-left.
        """
        ntotal = len(state.u_old)
        nghost = self.nghost

        # Define the range of physical (non-ghost) cells
        idx_range = range(nghost, ntotal - nghost)

        if self.config.dt_dx < 0:
            return reversed(idx_range)
        return idx_range

    # sweep() {{{2
    @abstractmethod
    def sweep(self, state: SolverState):
        pass


# SecondOrderImplicit() {{{1
@dataclass(frozen=True)
class SecondOrderImplicit(Scheme):
    nghost: int = 2
    config: SolverConfig = SolverConfig()

    # _update_cell_iter() {{{2
    def _update_cell_iter(
        self,
        u_new_i_current: float,
        state: SolverState,
        i: int,
    ) -> float:
        u_old = state.u_old
        u_new = state.u_new
        slope = state.slope

        dt_dx = self.config.dt_dx

        slope_i_current = compute_slope(
            state, i=i, u_new_i=u_new_i_current, config=self.config
        )

        # fmt: off
        i_upw = i-1 if dt_dx > 0 else i+1
        u_new_i_next = \
                (u_old[i] + abs(dt_dx) * u_new[i_upw]) / (1.0 + abs(dt_dx)) \
                - 0.5 * dt_dx * (slope_i_current - slope[i_upw])
        # fmt: on
        return u_new_i_next

    # sweep() {{{2
    def sweep(self, state: SolverState):
        if state.niter is None:
            state.niter = np.zeros_like(state.u_old, dtype=int)

        for i in self.cell_indices(state):
            u_new_i_guess = compute_guess(state, i=i, config=self.config)

            result = simple_fixed_point(
                self._update_cell_iter,
                u_new_i_guess,
                args=(state, i),
                tol=self.config.iteration.tol,
                maxiter=self.config.iteration.maxiter,
            )
            if result.success:
                state.u_new[i] = result.x
                state.slope[i] = compute_slope(state, i, result.x, self.config)
                state.niter[i] = result.nit

                # print(f"Cell {i} converged in {state.niter[i]} number of iterations.")
            else:
                print(f"Warning: Solver failed to converge at cell {i}.")
                print(f"Message: {result.message}")

                state.u_new[i] = u_new_i_guess


# HighResImplicit() {{{1
@dataclass(frozen=True)
class HighResImplicit(Scheme):
    nghost: int = 2
    config: SolverConfig = SolverConfig()

    # _update_cell_iter() {{{2
    def _update_cell_iter(
        self,
        u_new_i_current: float,
        state: SolverState,
        i: int,
    ) -> float:
        u_old = state.u_old
        u_new = state.u_new
        slope = state.slope
        eq = self.config.equation

        dt_dx = self.config.dt_dx

        slope_i_current = compute_slope(
            state, i=i, u_new_i=u_new_i_current, config=self.config
        )

        a_im1 = eq.speed(u_old[i - 1], u_new[i - 1])
        a_i = eq.speed(u_old[i], u_new_i_current)

        mu_im1 = dt_dx * a_im1
        c_im1 = mu_im1 * (1 + mu_im1) * 0.5 * slope[i - 1]

        mu_i = dt_dx * a_i
        c_i = mu_i * (1 + mu_i) * 0.5 * slope_i_current

        rhs = u_old[i] + dt_dx * eq.flux(u_new[i - 1]) - c_i + c_im1
        return solve_for_u(eq, rhs, dt_dx)

    # sweep() {{{2
    def sweep(self, state: SolverState):
        for i in self.cell_indices(state):
            u_new_i_guess = state.u_old[i]

            result = simple_fixed_point(
                self._update_cell_iter,
                u_new_i_guess,
                args=(state, i),
                tol=self.config.iteration.tol,
                maxiter=self.config.iteration.maxiter,
            )

            if result.success:
                state.u_new[i] = result.x
                state.slope[i] = compute_slope(state, i, result.x, self.config)
                state.niter[i] = result.nit

                # print(f"Cell {i} converged in {state.niter[i]} number of iterations.")
            else:
                print(f"Warning: Solver failed to converge at cell {i}.")
                print(f"Message: {result.message}")

                state.u_new[i] = u_new_i_guess


# Lozano() {{{1
@dataclass(frozen=True)
class Lozano(Scheme):
    nghost: int = 2
    config: SolverConfig = SolverConfig()

    # _flux_pos() {{{2
    def _flux_pos(self, u: float) -> float:
        eq = self.config.equation
        if u > 0.0:
            return eq.flux(u)
        return 0.0

    # _flux_neg() {{{2
    def _flux_neg(self, u: float) -> float:
        eq = self.config.equation
        if u > 0.0:
            return 0.0
        return eq.flux(u)

    # _update_cell() {{{2
    def _update_cell(self, state: SolverState, i: int, direction: str) -> float:
        u_old = state.u_old
        u_new = state.u_new
        dt_dx = self.config.dt_dx
        eq = self.config.equation

        if direction == "pos":
            rhs = u_old[i] + dt_dx * self._flux_pos(u_new[i - 1])
            if rhs > 0.0:
                return solve_for_u(eq, rhs, dt_dx, direction=direction)
            else:
                return rhs

        rhs = u_new[i] - dt_dx * self._flux_neg(u_new[i + 1])
        if rhs > 0.0:
            return rhs
        else:
            return solve_for_u(eq, rhs, dt_dx, direction=direction)


    # sweep() {{{2
    def sweep(self, state: SolverState):
        for i in self.cell_indices(state):
            state.u_new[i] = self._update_cell(state, i, direction="pos")
        for i in reversed(self.cell_indices(state)):
            state.u_new[i] = self._update_cell(state, i, direction="neg")
