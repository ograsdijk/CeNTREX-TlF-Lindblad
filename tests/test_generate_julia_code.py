from pathlib import Path
import pickle

import centrex_TlF_couplings as couplings
import centrex_TlF_lindblad as lindblad
import numpy as np
import pytest
import sympy as smp
from centrex_TlF_hamiltonian import hamiltonian, states


def test_generate_preamble():
    odepars = lindblad.odeParameters(Ωl=1.56e6, δl=0.0,)
    x_select = states.QuantumSelector(J=1)
    b_select = states.QuantumSelector(J=1, F=1, F1=1 / 2, P=1)
    transitions = [
        couplings.TransitionSelector(
            ground=1 * states.generate_coupled_states_X(x_select),
            excited=1 * states.generate_coupled_states_B(b_select),
            polarizations=[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
            polarization_symbols=smp.symbols("Plz Plx"),
            Ω=smp.Symbol("Ωl", comlex=True),
            δ=smp.Symbol("δl"),
            description="Q1 F1'=1/2 F'=1",
        )
    ]
    with pytest.raises(Exception):
        lindblad.generate_preamble(odepars, transitions)
    odepars = lindblad.odeParameters(
        Ωl=1.56e6,
        δl=0.0,
        ωp=2 * np.pi * 1e6,
        Pl="sin(ωp*t)",
        Plx="Pl >= 0",
        Plz="Pl < 0",
    )
    preamble = lindblad.generate_preamble(odepars, transitions)
    assert preamble == (
        "function Lindblad_rhs!(du, ρ, p, t)\n    \t@inbounds begin\n    "
        "\t\tΩl = p[1]\n\t\tδl = p[2]\n\t\tωp = p[3]\n\t\tPl = sin(ωp*t)"
        "\n\t\tPlx = Pl >= 0\n\t\tPlz = Pl < 0\n\t\tΩlᶜ = conj(Ωl)\n"
    )


def test_system_of_equations_to_lines():
    x_select = states.QuantumSelector(J=1)
    b_select = states.QuantumSelector(J=1, F=1, F1=1 / 2, P=1)
    transitions = [
        couplings.TransitionSelector(
            ground=1 * states.generate_coupled_states_X(x_select),
            excited=1 * states.generate_coupled_states_B(b_select),
            polarizations=[np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])],
            polarization_symbols=smp.symbols("Plz Plx"),
            Ω=smp.Symbol("Ωl", comlex=True),
            δ=smp.Symbol("δl"),
            description="Q1 F1'=1/2 F'=1",
        )
    ]
    H_reduced = hamiltonian.generate_total_reduced_hamiltonian(
        X_states_approx=list(states.generate_coupled_states_X(x_select)),
        B_states_approx=list(states.generate_coupled_states_B(b_select)),
    )
    QN = list(1 * states.generate_coupled_states_X(x_select)) + list(
        1 * states.generate_coupled_states_B(b_select)
    )
    coupl = []
    for transition in transitions:
        coupl.append(
            couplings.generate_coupling_field_automatic(
                transition.ground,
                transition.excited,
                QN,
                H_reduced.H_int,
                H_reduced.QN,
                H_reduced.V_ref_int,
                pol_vecs=transition.polarizations,
            )
        )
    H_symbolic = lindblad.generate_total_symbolic_hamiltonian(
        H_reduced.QN, H_reduced.H_int, coupl, transitions
    )
    C_array = couplings.collapse_matrices(
        QN,
        list(1 * states.generate_coupled_states_X(x_select)),
        list(1 * states.generate_coupled_states_B(b_select)),
        gamma=2 * np.pi * 1.56e6,
    )
    system = lindblad.generate_system_of_equations_symbolic(
        H_symbolic, C_array, fast=True
    )
    code_lines = lindblad.system_of_equations_to_lines(system)

    with open(
        Path(__file__).parent / "test_system_of_equations_to_lines.pkl", "rb"
    ) as f:
        system_test, code_lines_test = pickle.load(f)

    assert system == system_test
    assert code_lines == code_lines_test
