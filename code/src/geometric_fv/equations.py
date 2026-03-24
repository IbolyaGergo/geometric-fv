from abc import ABC, abstractmethod


# Equation {{{1
class Equation(ABC):
    # flux {{{2
    @abstractmethod
    def flux(self, u: float) -> float:
        """Compute the physical flux f(u)."""
        pass

    # dfdu {{{2
    @abstractmethod
    def dfdu(self, u: float) -> float:
        """Compute the physical speed f'(u)."""
        pass

    # speed {{{2
    def speed(self, u1: float, u2: float) -> float:
        """Compute the numerical wave speed."""
        du = u2 - u1
        if abs(du) < 1e-14:
            return self.dfdu(u1)
        return (self.flux(u2) - self.flux(u1)) / du

# Burgers {{{1
class Burgers(Equation):
    def flux(self, u: float) -> float:
        return 0.5 * u**2

    def dfdu(self, u: float) -> float:
        return u

# LinearAdvection {{{1
class LinearAdvection(Equation):
    def __init__(self, a: float = 1.0):
        self.a = a

    def flux(self, u: float) -> float:
        return self.a * u

    def dfdu(self, u: float) -> float:
        return self.a
