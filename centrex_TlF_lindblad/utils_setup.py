import logging
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Union

import centrex_TlF_couplings as couplings_TlF
import numpy as np
import numpy.typing as npt
import sympy as smp
from centrex_TlF_hamiltonian import hamiltonian, states
from julia import Main

from . import utils_decay as decay
from .generate_hamiltonian import generate_total_symbolic_hamiltonian
from .generate_julia_code import generate_preamble, system_of_equations_to_lines
from .generate_system_of_equations import generate_system_of_equations_symbolic
from .ode_parameters import odeParameters
from .utils_julia import generate_ode_fun_julia, initialize_julia
from .utils import SystemParameters

__all__ = ["generate_OBE_system", "setup_OBE_system_julia", "OBESystem"]


@dataclass
class OBESystem:
    ground: Sequence[states.State]
    excited: Sequence[states.State]
    QN: Sequence[states.State]
    H_int: npt.NDArray[np.complex_]
    V_ref_int: npt.NDArray[np.complex_]
    couplings: List[Any]
    H_symbolic: smp.matrices.dense.MutableDenseMatrix
    C_array: npt.NDArray[np.float_]
    system: smp.matrices.dense.MutableDenseMatrix
    code_lines: List[str]
    full_output: bool = False
    preamble: str = ""
    QN_original: Optional[Sequence[states.State]] = None
    decay_channels: Optional[Sequence[decay.DecayChannel]] = None


def generate_OBE_system(
    system_parameters: SystemParameters,
    transitions: Sequence[couplings_TlF.TransitionSelector],
    qn_compact: Optional[
        Union[states.QuantumSelector, Sequence[states.QuantumSelector]]
    ] = None,
    decay_channels: Optional[
        Union[Sequence[decay.DecayChannel], decay.DecayChannel]
    ] = None,
    E: npt.NDArray[np.float_] = np.array([0.0, 0.0, 0.0]),
    B: npt.NDArray[np.float_] = np.array([0.0, 0.0, 1e-5]),
    X_constants: hamiltonian.constants.XConstants = hamiltonian.XConstants(),
    B_constants: hamiltonian.constants.BConstants = hamiltonian.BConstants(),
    nuclear_spins: states.TlFNuclearSpins = states.TlFNuclearSpins(),
    Jmin_X: Optional[int] = None,
    Jmax_X: Optional[int] = None,
    Jmin_B: Optional[int] = None,
    Jmax_B: Optional[int] = None,
    transform: Optional[npt.NDArray[np.complex_]] = None,
    H_func_X: Optional[Callable] = None,
    H_func_B: Optional[Callable] = None,
    verbose: bool = False,
) -> OBESystem:
    """Convenience function for generating the symbolic OBE system of equations
    and Julia code.

    Args:
        system_parameters (SystemParameters): dataclass holding system parameters

        transitions (list): list of TransitionSelectors defining the transitions
                            used in the OBE system.
        qn_compact (QuantumSelector): dataclass specifying a subset of states to
                                        select based on the quantum numbers
        decay_channels (DecayChannel): dataclass specifying the decay channel to
                                        add
        verbose (bool, optional): Log progress to INFO. Defaults to False.

    Returns:
        OBESystem: dataclass designed to hold the generated values
                    ground, exxcited, QN, H_int, V_ref_int, couplings, H_symbolic,
                    C_array, system, code_lines
    """
    QN_X_original = list(states.generate_coupled_states_X(system_parameters.X))
    QN_B_original = list(states.generate_coupled_states_B(system_parameters.B))
    QN_original = QN_X_original + QN_B_original
    rtol = None
    if verbose:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        logger.info("generate_OBE_system: 1/6 -> Generating the reduced Hamiltonian")
    H_reduced = hamiltonian.generate_total_reduced_hamiltonian(
        X_states_approx=QN_X_original,
        B_states_approx=QN_B_original,
        E=E,
        B=B,
        Jmin_X=Jmin_X,
        Jmax_X=Jmax_X,
        Jmin_B=Jmin_B,
        Jmax_B=Jmax_B,
        rtol=rtol,
        X_constants=X_constants,
        B_constants=B_constants,
        nuclear_spins=nuclear_spins,
        transform=transform,
        H_func_X=H_func_X,
        H_func_B=H_func_B,
    )
    ground_states = H_reduced.X_states
    excited_states = H_reduced.B_states
    QN = H_reduced.QN
    H_int = H_reduced.H_int

    V_ref_int = H_reduced.V_ref_int
    if verbose:
        logger.info(
            "generate_OBE_system: 2/6 -> "
            "Generating the couplings corresponding to the transitions"
        )
    couplings = []
    for transition in transitions:
        if transition.ground_main is not None and transition.excited_main is not None:
            couplings.append(
                couplings_TlF.generate_coupling_field(
                    transition.ground_main,
                    transition.excited_main,
                    transition.ground,
                    transition.excited,
                    QN_original,
                    H_int,
                    QN,
                    V_ref_int,
                    pol_vecs=transition.polarizations,
                    pol_main=transition.polarizations[0],
                )
            )
        else:
            couplings.append(
                couplings_TlF.generate_coupling_field_automatic(
                    transition.ground,
                    transition.excited,
                    QN_original,
                    H_int,
                    QN,
                    V_ref_int,
                    pol_vecs=transition.polarizations,
                )
            )

    if verbose:
        logger.info("generate_OBE_system: 3/6 -> Generating the symbolic Hamiltonian")
    if qn_compact is not None:
        H_symbolic, QN_compact = generate_total_symbolic_hamiltonian(
            QN, H_int, couplings, transitions, qn_compact=qn_compact
        )
    else:
        H_symbolic = generate_total_symbolic_hamiltonian(
            QN, H_int, couplings, transitions
        )

    if verbose:
        logger.info("generate_OBE_system: 4/6 -> Generating the collapse matrices")
    C_array = couplings_TlF.collapse_matrices(
        QN,
        ground_states,
        excited_states,
        gamma=system_parameters.Γ,
        qn_compact=qn_compact,
    )

    if decay_channels is not None:
        if isinstance(decay_channels, list):
            _decay_channels = decay_channels
        elif not isinstance(decay_channels, (tuple, list, np.ndarray)):
            _decay_channels = [decay_channels]
        elif isinstance(decay_channels, (tuple, np.ndarray)):
            _decay_channels = list(decay_channels)
        else:
            raise AssertionError(
                f"decay_channels is type f{type(decay_channels)}; supply a list, tuple"
                " or np.ndarray"
            )

        if qn_compact is not None:
            indices, H_symbolic = decay.add_levels_symbolic_hamiltonian(
                H_symbolic, _decay_channels, QN_compact, excited_states
            )
            QN_compact = decay.add_states_QN(_decay_channels, QN_compact, indices)
            C_array = decay.add_decays_C_arrays(
                _decay_channels, indices, QN_compact, C_array, system_parameters.Γ
            )
        else:
            indices, H_symbolic = decay.add_levels_symbolic_hamiltonian(
                H_symbolic, _decay_channels, QN, excited_states
            )
            QN = decay.add_states_QN(_decay_channels, QN_compact, indices)
            C_array = decay.add_decays_C_arrays(
                _decay_channels, indices, QN, C_array, system_parameters.Γ
            )
    if verbose:
        logger.info(
            "generate_OBE_system: 5/6 -> Transforming the Hamiltonian and collapse "
            "matrices into a symbolic system of equations"
        )
    system = generate_system_of_equations_symbolic(H_symbolic, C_array, fast=True)
    if verbose:
        logger.info(
            "generate_OBE_system: 6/6 -> Generating Julia code representing the system "
            "of equations"
        )
        logging.basicConfig(level=logging.WARNING)
    code_lines = system_of_equations_to_lines(system)
    obe_system = OBESystem(
        QN=QN_compact if qn_compact is not None else QN,
        ground=ground_states,
        excited=excited_states,
        couplings=couplings,
        H_symbolic=H_symbolic,
        H_int=H_int,
        V_ref_int=V_ref_int,
        C_array=C_array,
        system=system,
        code_lines=code_lines,
        QN_original=None if qn_compact is None else QN,
        decay_channels=_decay_channels if decay_channels else None,
    )
    return obe_system


def setup_OBE_system_julia(
    system_parameters: SystemParameters,
    ode_parameters: odeParameters,
    transitions: Sequence[couplings_TlF.TransitionSelector],
    qn_compact: Optional[
        Union[Sequence[states.QuantumSelector], states.QuantumSelector]
    ] = None,
    full_output: bool = False,
    decay_channels: Optional[
        Union[Sequence[decay.DecayChannel], decay.DecayChannel]
    ] = None,
    E: npt.NDArray[np.float_] = np.array([0.0, 0.0, 0.0]),
    B: npt.NDArray[np.float_] = np.array([0.0, 0.0, 1e-5]),
    X_constants: hamiltonian.constants.XConstants = hamiltonian.XConstants(),
    B_constants: hamiltonian.constants.BConstants = hamiltonian.BConstants(),
    nuclear_spins: states.TlFNuclearSpins = states.TlFNuclearSpins(),
    Jmin_X: Optional[int] = None,
    Jmax_X: Optional[int] = None,
    Jmin_B: Optional[int] = None,
    Jmax_B: Optional[int] = None,
    transform: Optional[npt.NDArray[np.complex_]] = None,
    H_func_X: Optional[Callable] = None,
    H_func_B: Optional[Callable] = None,
    verbose: bool = False,
    init_julia: bool = True,
):
    """Convenience function for generating the OBE system and initializing it in
    Julia

    Args:
        system_parameters (SystemParameters): dataclass holding the system
                                                parameters, e.g. Γ,
                                                (laser) ground states,
                                                (laser) excited states
        ode_parameters (odeParameters): dataclass containing the ode parameters.
                                        e.g. Ω, δ, vz, ..., etc.
        transitions (TransitionSelector): object containing all information
                                            required to generate the coupling
                                            matrices and symbolic matrix for
                                            each transition
        qn_compact (QuantumSelector): dataclass specifying a subset of states to
                                        select based on the quantum numbers
        full_output (bool, optional): Returns all matrices, states etc. if True,
                                        Returns only QN if False.
                                        Defaults to False.
        decay_channels (DecayChannel): dataclass specifying the decay channel to
                                        add
        verbose (bool, optional): Log progress to INFO. Defaults to False.

    Returns:
        full_output == True:
            list: list of states in system
        full_output == False:
            OBESystem: dataclass designed to hold the generated values
                        ground, exxcited, QN, H_int, V_ref_int, couplings,
                        H_symbolic, C_array, system, code_lines, preamble
    """
    obe_system = generate_OBE_system(
        system_parameters,
        transitions,
        qn_compact=qn_compact,
        decay_channels=decay_channels,
        E=E,
        B=B,
        X_constants=X_constants,
        B_constants=B_constants,
        nuclear_spins=nuclear_spins,
        Jmin_X=Jmin_X,
        Jmax_X=Jmax_X,
        Jmin_B=Jmin_B,
        Jmax_B=Jmax_B,
        transform=transform,
        H_func_X=H_func_X,
        H_func_B=H_func_B,
        verbose=verbose,
    )
    obe_system.full_output = full_output
    if verbose:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        logger.info("setup_OBE_system_julia: 1/3 -> Generating the preamble")
    obe_system.preamble = generate_preamble(ode_parameters, transitions)

    if init_julia:
        if verbose:
            logger.info(
                "setup_OBE_system_julia: 2/3 -> Initializing Julia on "
                f"{system_parameters.nprocs} cores"
            )
        initialize_julia(nprocs=system_parameters.nprocs)

    if verbose:
        logger.info(
            "setup_OBE_system_julia: 3/3 -> Defining the ODE equation and parameters in"
            " Julia"
        )
        logging.basicConfig(level=logging.WARNING)
    generate_ode_fun_julia(obe_system.preamble, obe_system.code_lines)
    Main.eval(f"@everywhere Γ = {system_parameters.Γ}")
    ode_parameters.generate_p_julia()
    if not full_output:
        return obe_system.QN
    else:
        return obe_system
