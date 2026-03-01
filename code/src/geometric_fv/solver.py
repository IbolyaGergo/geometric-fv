from dataclasses import dataclass

import numpy as np


@dataclass
class SolverState:
    u_old: np.ndarray
    u_new: np.ndarray
    slope: np.ndarray
    cfl: float
