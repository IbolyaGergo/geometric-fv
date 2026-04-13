from typing import Callable, NamedTuple

import numpy as np


# Fixed-Point Iteration {{{1
# FixedPointResult {{{2
class FixedPointResult(NamedTuple):
    x: float
    success: bool
    nit: int
    message: str


# simple_fixed_point() {{{2
def simple_fixed_point(
    func: Callable, x0: float, args=(), tol: float = 1e-8, maxiter: int = 50
) -> FixedPointResult:
    """
    Finds a solution for x = func(x, *args) using fixed-point iteration.

    Returns:
        FixedPointResult: A namedtuple containing the result and solver info.
    """
    x_old = x0
    for i in range(maxiter):
        x_new = func(x_old, *args)
        if abs(x_new - x_old) < tol:
            return FixedPointResult(
                x=x_new,
                success=True,
                nit=i + 1,  # Number of iterations
                message="Converged successfully.",
            )
        x_old = x_new

    # If the loop completes without converging, return a failure state
    return FixedPointResult(
        x=x_new,  # Return the last computed value
        success=False,
        nit=maxiter,
        message=f"Failed to converge after {maxiter} iterations.",
    )


# Error Analysis {{{1
# calculate_norms() {{{2
def calculate_norms(
    numerical: np.ndarray, exact: np.ndarray, dx: float
) -> dict[str, float]:
    """Calculates L1, L2 and Linf error norms."""
    diff = np.abs(numerical - exact)

    l1 = np.sum(diff * dx)
    l2 = np.sqrt(np.sum(diff**2 * dx))
    linf = np.max(diff)

    return {"L1": l1, "L2": l2, "Linf": linf}
