from typing import Any, Optional, Union, List
from dataclasses import dataclass
import sympy as smp
import numpy.typing as npt
from centrex_TlF_hamiltonian import states

__all__ = ["generate_density_matrix_symbolic", "SystemParameters"]


def recursive_subscript(i: int) -> str:
    # chr(0x2080+i) is unicode for
    # subscript num(i), resulting in x₀₀ for example
    if i < 10:
        return chr(0x2080 + i)
    else:
        return recursive_subscript(i // 10) + chr(0x2080 + i % 10)


def generate_density_matrix_symbolic(
    levels: int,
) -> smp.matrices.dense.MutableDenseMatrix:
    ρ = smp.zeros(levels, levels)
    levels = levels
    for i in range(levels):
        for j in range(i, levels):
            # \u03C1 is unicode for ρ,
            if i == j:
                ρ[i, j] = smp.Symbol(
                    u"\u03C1{0},{1}".format(
                        recursive_subscript(i), recursive_subscript(j)
                    )
                )
            else:
                ρ[i, j] = smp.Symbol(
                    u"\u03C1{0},{1}".format(
                        recursive_subscript(i), recursive_subscript(j)
                    )
                )
                ρ[j, i] = smp.Symbol(
                    u"\u03C1{1},{0}".format(
                        recursive_subscript(i), recursive_subscript(j)
                    )
                )
    return ρ


@dataclass
class SystemParameters:
    nprocs: int
    Γ: float
    X: Optional[
        Union[states.QuantumSelector, List[states.QuantumSelector], npt.NDArray[Any]]
    ] = None
    B: Optional[
        Union[states.QuantumSelector, List[states.QuantumSelector], npt.NDArray[Any]]
    ] = None
