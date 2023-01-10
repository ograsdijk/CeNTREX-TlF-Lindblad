import pickle
from pathlib import Path

import centrex_tlf_couplings as couplings
import numpy as np
import pytest
import sympy as smp
from centrex_tlf_hamiltonian import hamiltonian, states, transitions

import centrex_tlf_lindblad as lindblad


def test_generate_qn_compact():

    trans = [
        transitions.OpticalTransition(transitions.OpticalTransitionType.R, 0, 3 / 2, 1)
    ]
    transition_selectors = couplings.generate_transition_selectors(
        trans, [[couplings.polarization_Z]]
    )

    try:
        with open(Path(__file__).parent / "test_generate_qn_compact.pkl", "rb") as f:
            H_reduced = pickle.load(f)
    except:
            H_reduced = hamiltonian.generate_reduced_hamiltonian_transitions(trans)
    qn_compact = lindblad.utils_compact.generate_qn_compact(trans, H_reduced)
    assert qn_compact == [
        states.QuantumSelector(J=2, electronic=states.ElectronicState.X)
    ]

def test_compact_symbolic_hamiltonian_indices():
    hamiltonian = smp.zeros(5)
    indices = np.array([2,3])
    arr = lindblad.utils_compact.compact_symbolic_hamiltonian_indices(
        hamiltonian, indices
    )
    assert arr.shape == (4,4)

    hamiltonian = smp.ones(5)
    indices = np.array([2,3])
    with pytest.raises(AssertionError):
        arr = lindblad.utils_compact.compact_symbolic_hamiltonian_indices(
            hamiltonian, indices
        )


