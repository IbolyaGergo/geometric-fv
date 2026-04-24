from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SolverState:
    u_old: np.ndarray
    u_new: np.ndarray
    slope: np.ndarray
    niter: np.ndarray
    speed: np.ndarray
    flux: np.ndarray
