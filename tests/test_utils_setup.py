import pickle
from pathlib import Path

import centrex_TlF_couplings as couplings
import centrex_TlF_lindblad as lindblad
import numpy as np
import sympy as smp
from centrex_TlF_hamiltonian import states


def test_generate_OBE_system():
    ground = [
        states.QuantumSelector(J=1, electronic=states.ElectronicState.X),
        states.QuantumSelector(J=3, electronic=states.ElectronicState.X),
    ]
    excited = states.QuantumSelector(
        J=1, F=1, F1=1 / 2, P=+1, electronic=states.ElectronicState.B
    )
    syspars = lindblad.SystemParameters(
        nprocs=2, Γ=2 * np.pi * 1.56e6, X=ground, B=excited
    )
    odepars = lindblad.odeParameters(
        Ωl=2 * np.pi * 1.56e6,
        δl=0,
        ωp=2 * np.pi * 1.56e6,
        φp=0.0,
        Pl="sin(ωp*t + φp)",
        Plz="Pl>0",
        Plx="Pl<=0",
        y0=0,
        vz=184,
        vy=0,
    )

    transitions = [
        couplings.TransitionSelector(
            ground=1 * states.generate_coupled_states_X(ground[0]),
            excited=1 * states.generate_coupled_states_B(excited),
            polarizations=[np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])],
            polarization_symbols=smp.symbols("Plx Plz"),
            Ω=smp.Symbol("Ωl", complex=True),
            δ=smp.Symbol("δl"),
            description="Q1 F1'=1/2 F'=1",
        )
    ]

    decay_channels = [
        lindblad.DecayChannel(
            ground=1
            * states.CoupledBasisState(
                None, None, None, None, None, None, v="other", P=-1
            ),
            excited=excited,
            branching=1e-2,
            description="vibrational decay",
        )
    ]
    obe_system = lindblad.generate_OBE_system(
        system_parameters=syspars,
        transitions=transitions,
        verbose=True,
        qn_compact=states.QuantumSelector(J=3, electronic=states.ElectronicState.X),
        decay_channels=decay_channels,
    )
    with open(Path(__file__).parent / "C_array.npy", "rb") as f:
        C_array = np.load(f)
    assert np.allclose(C_array, obe_system.C_array, rtol=1e-3)

    with open(Path(__file__).parent / "obe_system.pkl", "rb") as f:
        obe_truth = pickle.load(f)
    assert obe_system.QN == obe_truth.QN
    assert np.allclose(obe_system.H_int, obe_truth.H_int)
    assert obe_system.H_symbolic == obe_truth.H_symbolic
    assert obe_system.code_lines == obe_truth.code_lines
    assert obe_system.system == obe_system.system
