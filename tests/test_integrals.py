import numpy as np
import pytest
from src.integrals import (
    compute_kinetic_matrix, compute_overlap_matrix,
    compute_nuclear_attraction_matrix, compute_core_hamiltonian,
    compute_eri_tensor
)

@pytest.fixture
def h2_data():
    primitives_raw = [[
        (3.42525091, 0.15432897),
        (0.62391373, 0.53532814),
        (0.16885540, 0.44463454)],
        [
        (3.42525091, 0.15432897),
        (0.62391373, 0.53532814),
        (0.16885540, 0.44463454)]
    ]
    primitives = []
    for shell in primitives_raw:
        normalized_shell = []
        for alpha, coeff in shell:
            norm_factor = (2.0 * alpha / np.pi) ** 0.75
            normalized_shell.append((alpha, coeff * norm_factor))
        primitives.append(normalized_shell)

    pos = [0.0, 1.4]
    Z = 1.0
    return primitives, pos, Z

@pytest.fixture
def heh_plus_data():
    # STO-3G basis for HeH+ (He: Z=2, H: Z=1)
    he_primitives = [
        (6.36242139, 0.15432897),
        (1.15892300, 0.53532814),
        (0.31364979, 0.44463454)
    ]
    h_primitives = [
        (3.42525091, 0.15432897),
        (0.62391373, 0.53532814),
        (0.16885540, 0.44463454)
    ]

    primitives = []
    for shell in primitives_raw:
        normalized_shell = []
        for alpha, coeff in shell:
            norm_factor = (2.0 * alpha / np.pi) ** 0.75
            normalized_shell.append((alpha, coeff * norm_factor))
        primitives.append(normalized_shell)

    pos = [0.0, 1.4632]  # Equilibrium distance for HeH+
    Z_he, Z_h = 2.0, 1.0
    return primitives, pos, Z_he, Z_h

# def test_overlap_matrix_values(h2_data):
#     primitives, pos, _ = h2_data
#     S = compute_overlap_matrix(primitives, pos)
#     expected = np.array([[1.0, 0.6593],
#                          [0.6593, 1.0]])
#     np.testing.assert_allclose(S, expected, rtol=1e-3)

def test_core_hamiltonian(h2_data):
    primitives_h2, pos, Z = h2_data
    primitives_hehp, pos, Z = heh_plus_data
    T_1 = compute_kinetic_matrix(primitives_h2, pos)
    T_2 = compute_kinetic_matrix(primitives_hehp, pos)
    V_1 = compute_nuclear_attraction_matrix(primitives_h2, pos, Z)
    V_2 = compute_nuclear_attraction_matrix(primitives_hehp, pos, Z)
    H_core_h2 = compute_core_hamiltonian(T_1, V_1)
    H_core_hehp = compute_core_hamiltonian(T_2, V_2)
    expected_h2 = np.array([[-1.1204, -0.9584],
                         [-0.9584, -1.1204]])
    expected_hehp = np.array([[-2.6525, -1.3466],
                         [-1.3466, -1.7318]])
    np.testing.assert_allclose(H_core_h2, expected_h2, rtol=1e-3)
    np.testing.assert_allclose(H_core_hehp, expected_hehp, rtol=1e-3)


def test_eri_tensor_values(h2_data):
    primitives, pos, _ = h2_data
    eri = compute_eri_tensor(primitives, pos)

    assert eri.shape == (2, 2, 2, 2)

    expected_values = {
        (0, 0, 0, 0): 0.7746,
        (1, 1, 1, 1): 0.7746,
        (0, 0, 1, 1): 0.5697,
        (1, 0, 0, 0): 0.4441,
        (1, 1, 1, 0): 0.4441,
        (1, 0, 1, 0): 0.2970
    }

    for idx, val in expected_values.items():
        i, j, k, l = idx
        np.testing.assert_allclose(eri[i, j, k, l], val, rtol=1e-3)

