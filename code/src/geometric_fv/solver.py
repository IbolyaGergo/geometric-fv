from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geometric_fv.equations import Burgers, Equation, LinearAdvection


@dataclass
class SolverState:
    u_old: np.ndarray
    u_new: np.ndarray
    slope: np.ndarray
    niter: np.ndarray
    speed: np.ndarray
    flux: np.ndarray
