import numpy as np
from scf import scf_rhf
import time

class Primitive:
    def __init__(self, zeta, coeff, center, angmom):
        self.zeta = zeta
        self.coeff = coeff
        self.center = np.array(center)
        self.angmom = angmom

class BasisFunction:
    def __init__(self, primitives):
        self.primitives = [Primitive(zeta, coeff, center, angmom)
                           for zeta, coeff, center, angmom in primitives]

def build_C2H4_sto3g_basis():
    basis_set = {}
    idx = 0
    bohr = 1.0 / 0.529177210903

    # === Atom coordinates in Angstrom ===
    C_positions = [
        [-0.6695, 0.0000, 0.0000],
        [ 0.6695, 0.0000, 0.0000],
    ]
    H_positions = [
        [-1.2321,  0.9289, 0.0000],
        [-1.2321, -0.9289, 0.0000],
        [ 1.2321,  0.9289, 0.0000],
        [ 1.2321, -0.9289, 0.0000],
    ]

    C_positions = [[x * bohr for x in r] for r in C_positions]
    H_positions = [[x * bohr for x in r] for r in H_positions]

    # === STO-3G Parameters ===
    C_S_exps = [71.61683735, 13.04509632, 3.530512160]
    C_S_coeffs = [0.1543289673, 0.5353281423, 0.4446345422]

    C_SP_exps = [2.941249355, 0.6834830964, 0.2222899159]
    C_SP_s_coeffs = [-0.09996722919, 0.3995128261, 0.7001154689]
    C_SP_p_coeffs = [0.1559162750, 0.6076837186, 0.3919573931]

    H_exps = [3.42525091, 0.62391373, 0.16885540]
    H_coeffs = [0.15432897, 0.53532814, 0.44463454]

    # === Carbon atoms ===
    for C_pos in C_positions:
        C_1s = BasisFunction([
            (exp, coeff, C_pos, (0, 0, 0))
            for exp, coeff in zip(C_S_exps, C_S_coeffs)
        ])
        basis_set[idx] = C_1s
        idx += 1

        C_2s = BasisFunction([
            (exp, coeff, C_pos, (0, 0, 0))
            for exp, coeff in zip(C_SP_exps, C_SP_s_coeffs)
        ])
        basis_set[idx] = C_2s
        idx += 1

        for angmom in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
            func = BasisFunction([
                (exp, coeff, C_pos, angmom)
                for exp, coeff in zip(C_SP_exps, C_SP_p_coeffs)
            ])
            basis_set[idx] = func
            idx += 1

    # === Hydrogens ===
    for H_pos in H_positions:
        H_1s = BasisFunction([
            (exp, coeff, H_pos, (0, 0, 0))
            for exp, coeff in zip(H_exps, H_coeffs)
        ])
        basis_set[idx] = H_1s
        idx += 1

    return basis_set, C_positions + H_positions, [6.0, 6.0] + [1.0]*4

basis_set, R_nuc, Z_nuc = build_C2H4_sto3g_basis()
nuclei = list(zip(Z_nuc, R_nuc))

start_time = time.time()
scf_rhf(n_elec=12, R_nuc=R_nuc, Z_nuc=Z_nuc, verbose=1,
        basis_set=basis_set, nuclei=nuclei)
end_time = time.time()

print(f"Total runtime: {end_time - start_time:.4f} seconds")
