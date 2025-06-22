import numpy as np
import pytest
from scipy.linalg import fractional_matrix_power
from src.integrals import build_integral_arrays
from src.scf import scf_rhf, build_fock_matrix

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
    primitives = [he_primitives, h_primitives]
    pos = [0.0, 1.4632]  # Equilibrium distance for HeH+
    Z_he, Z_h = 2.0, 1.0
    return primitives, pos, Z_he, Z_h

# H2_ref = {
#     'energy_electronic': -1.7832761512906234,
#     'energy_nuclear': 1.7142857142857144,
#     'energy_total': -0.06899043700490914,
#     'G_matrix': [[-0.51039083  0.28844484],
#           [-0.52466038 -0.25695278]]
#     'S_matrix':  [[1.         0.65931901],
#            [0.65931901 1.        ]]
# }

# HeHp_ref = {
#     'energy_electronic': -4.285947819745563,
#     'energy_nuclear': 2.7142857142857144,
#     'energy_total': -1.5716621054598487,
#     'G_matrix':  [[-1.49096667 -0.37245117],
#            [-1.11678035 -0.25326028]]
#     'S_matrix': [[1.         0.47316108],
#           [0.47316108 1.        ]]
# }


if __name__ == "__main__":

    primitives_lithium = [
        build_sto3g_basis_2s(6.3, shell="1s"),
        build_sto3g_basis_2s(1.0, shell="2s"),
    ]

    primitives_hydrogen = [
        build_sto3g_basis_2s(1.0, shell="1s"),
    ]

    primitives_lih = primitives_lithium + primitives_hydrogen

    pos_lih = np.array([[0,0,0],[0,0,0],[3.014,0,0]])
    # pos = [
    #     np.array([0.0]),  # Li
    #     np.array([0.0]),  # Li again (for 2s)
    #     np.array([1.6]),  # H
    # ]
    
    Z_nuc = [3,1]
    R_nuc = np.array([[0.0, 0.0, 0.0], [3.014, 0.0, 0.0]])
    Z_lih = (3.0, 3.0, 1.0)  # match order of basis centers
    n_elec_lih = 4


    S, H_core, eri_dict = build_integral_arrays(primitives=primitives_lih, pos=pos_lih, Z=Z_lih)

    print(S)






