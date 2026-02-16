import abc
from dataclasses import dataclass


class Scheme(abc.ABC):
    nghost: int

    def sweep(self, u_old, u_new, cfl):
        pass


@dataclass(frozen=True)
class ImplicitUpwind(Scheme):
    nghost: int = 1

    def sweep(self, u_old, u_new, cfl):
        nghost = self.nghost
        if cfl > 0:
            for i in range(nghost, len(u_old) - nghost):
                u_new[i] = (u_old[i] + cfl * u_new[i - 1]) / (1.0 + cfl)
        else:
            for i in reversed(range(nghost, len(u_old) - nghost)):
                u_new[i] = (u_old[i] + (-cfl) * u_new[i + 1]) / (1.0 + (-cfl))


@dataclass(frozen=True)
class Box(Scheme):
    nghost: int = 1

    def sweep(self, u_old, u_new, cfl):
        nghost = self.nghost
        coeff = (1 - cfl) / (1 + cfl)
        for i in range(nghost, len(u_old) - nghost):
            u_new[i] = coeff * u_old[i] + u_old[i - 1] - coeff * u_new[i - 1]
