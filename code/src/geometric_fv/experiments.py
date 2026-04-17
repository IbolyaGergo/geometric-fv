"""Registry for numerical experiments."""

from dataclasses import dataclass, field
from typing import Any, Type

from geometric_fv.enums import AvgSpeedType, LimiterType, SlopeType
from geometric_fv.schemes import (
    BoxBurgers,
    HighResImplicit,
    Lozano,
    Scheme,
)


@dataclass(frozen=True)
class StudyProfile:
    """Blueprint for a scheme experiment in a convergence study."""

    scheme_class: Type[Scheme]
    default_dt_dx: float = 1.8
    # We use default_factory=dict to ensure every instance gets its own
    # dict object. In Python, using a literal {} as a default value would
    # share the same dict across all instances.
    reconst_kwargs: dict[str, Any] = field(default_factory=dict)
    iteration_kwargs: dict[str, Any] = field(default_factory=dict)


STUDY_REGISTRY: dict[str, StudyProfile] = {
    "burgers-implup": StudyProfile(
        scheme_class=Lozano,
    ),
    "burgers-highres-full": StudyProfile(
        scheme_class=HighResImplicit,
        reconst_kwargs={"limiter_type": LimiterType.FULL},
    ),
}
