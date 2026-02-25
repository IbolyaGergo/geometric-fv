from dataclasses import dataclass

import numpy as np


@dataclass
class SolverState:
    u_old: np.ndarray
    u_new: np.ndarray
    slope: np.ndarray
    cfl: float

    def get_u_new(self, i: int, override: float = np.nan):
        return override if not np.isnan(override) else self.u_new[i]
    def get_slope(self, i: int, override: float = np.nan):
        return override if not np.isnan(override) else self.slope[i]
