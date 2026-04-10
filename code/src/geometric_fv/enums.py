from enum import Enum


class BCType(Enum):
    CONSTANT_EXTEND = "constant_extend"
    QUASI_PERIODIC = "quasi_periodic"


class SlopeType(Enum):
    BOX = 0


class LimiterType(Enum):
    FULL = 0
    NONE = 1
    TVD = 2
    TVD_SUFF = 3


class GuessType(Enum):
    IMPLICIT_UPWIND = 0
    BOX = 1


class AvgSpeedType(Enum):
    IMPLICIT_UPWIND = 0
