from dataclasses import dataclass

import numpy as np


@dataclass
class SolverState:
    u_old: np.ndarray
    u_new: np.ndarray
    slope: np.ndarray
    cfl: float

    def get_u_old(self, i: int):
        return self.u_old[i]
    def get_u_new(self, i: int, override: float = np.nan):
        return override if not np.isnan(override) else self.u_new[i]
    def get_slope(self, i: int, override: float = np.nan):
        return override if not np.isnan(override) else self.slope[i]
    def get_cfl(self):
        return self.cfl
    def get_len(self):
        return len(self.u_old)

    def set_u_new(self, i: int, val: float):
        self.u_new[i] = val
    def set_slope(self, i: int, val: float):
        self.slope[i] = val
