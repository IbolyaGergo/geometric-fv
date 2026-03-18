from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SolverState:
    u_old: np.ndarray
    u_new: np.ndarray
    slope: np.ndarray
    dt_dx: float
    niter: np.ndarray | None = None
    speed: np.ndarray | None = None
