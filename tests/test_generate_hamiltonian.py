import numpy as np
import sympy as smp
from centrex_tlf_hamiltonian import states, hamiltonian
import centrex_tlf_couplings as couplings
import centrex_tlf_lindblad as lindblad


def test_generate_symbolic_hamiltonian():
    x_select = states.QuantumSelector(J=1)
    b_select = states.QuantumSelector(J=1, F=1, F1=1 / 2, P=1)
    transitions = [
        couplings.TransitionSelector(
            ground=1 * states.generate_coupled_states_X(x_select),
            excited=1 * states.generate_coupled_states_B(b_select),
            polarizations=[np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])],
            polarization_symbols=smp.symbols("Plx Plz"),
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
    hamiltonian_symbolic = lindblad.generate_symbolic_hamiltonian(
        QN=H_reduced.QN,
        H_int=H_reduced.H_int,
        couplings=coupl,
        Ωs=[smp.Symbol("Ωl", complex=True)],
        δs=[smp.Symbol("δl")],
        pols=[[smp.Symbol("Plx"), smp.Symbol("Plz")]],
    )
    δl = smp.Symbol("δl")
    true_values = [
        1.0 * δl - 1336622.0309906,
        1.0 * δl - 1196891.62820435,
        1.0 * δl - 1196891.63879395,
        1.0 * δl - 1196891.64953613,
        1.0 * δl - 91349.8861083984,
        1.0 * δl - 91349.8859710693,
        1.0 * δl - 91349.8856658936,
        1.0 * δl,
        1.0 * δl - 0.0103759765625,
        1.0 * δl - 0.0206146240234375,
        1.0 * δl - 0.03094482421875,
        1.0 * δl - 0.041229248046875,
        0,
        -2.82593441009521,
        -5.65186500549316,
    ]
    for dh, dtv in zip(np.diag(hamiltonian_symbolic), true_values):
        assert np.abs(dh - dtv) <= 1e-8


def test_generate_total_symbolic_hamiltonian():
    x_select = states.QuantumSelector(J=1)
    x_select_compact = states.QuantumSelector(J=3, electronic=states.ElectronicState.X)
    b_select = states.QuantumSelector(J=1, F=1, F1=1 / 2, P=1)
    transitions = [
        couplings.TransitionSelector(
            ground=1 * states.generate_coupled_states_X(x_select),
            excited=1 * states.generate_coupled_states_B(b_select),
            polarizations=[np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])],
            polarization_symbols=smp.symbols("Plx Plz"),
            Ω=smp.Symbol("Ωl", comlex=True),
            δ=smp.Symbol("δl"),
            description="Q1 F1'=1/2 F'=1",
        )
    ]
    H_reduced = hamiltonian.generate_total_reduced_hamiltonian(
        X_states_approx=list(states.generate_coupled_states_X(x_select))
        + list(states.generate_coupled_states_X(x_select_compact)),
        B_states_approx=list(states.generate_coupled_states_B(b_select)),
    )
    QN = (
        list(1 * states.generate_coupled_states_X(x_select))
        + list(states.generate_coupled_states_X(x_select_compact))
        + list(1 * states.generate_coupled_states_B(b_select))
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
    hamiltonian_symbolic, QN_compact = lindblad.generate_total_symbolic_hamiltonian(
        QN=H_reduced.QN,
        H_int=H_reduced.H_int,
        couplings=coupl,
        transitions=transitions,
        qn_compact=x_select_compact,
    )
    assert hamiltonian_symbolic.shape == (16, 16)
    assert states.QuantumSelector(J=3, electronic=states.ElectronicState.X).get_indices(
        QN_compact
    ) == np.array([12])

    δl = smp.Symbol("δl")
    true_values = [
        1.0 * δl - 1336622.0309906,
        1.0 * δl - 1196891.62820435,
        1.0 * δl - 1196891.63879395,
        1.0 * δl - 1196891.64953613,
        1.0 * δl - 91349.8861083984,
        1.0 * δl - 91349.8859710693,
        1.0 * δl - 91349.8856658936,
        1.0 * δl,
        1.0 * δl - 0.0103759765625,
        1.0 * δl - 0.0206146240234375,
        1.0 * δl - 0.03094482421875,
        1.0 * δl - 0.041229248046875,
        502704902817.902,
        0,
        -2.82593441009521,
        -5.65186500549316,
    ]
    for dh, dtv in zip(np.diag(hamiltonian_symbolic), true_values):
        _dtv = float(dtv.subs(δl, 0)) if not isinstance(dtv, (int, float)) else dtv
        _dtv = 1.0 if _dtv == 0 else _dtv
        assert abs(dh - dtv) / abs(_dtv) < 1e-2
