import numpy as np
import time

from src.scf.rhf_os import scf_rhf

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

def build_H2O_sto3g_basis():
    basis_set = {}
    idx = 0

    # === Oxygen ===
    O_pos = [0.0, 0.0, 0.0]

    # 1s (contracted)
    O_S_exps = [130.7093214, 23.80886605, 6.443608313]
    O_S_coeffs = [0.1543289673, 0.5353281423, 0.4446345422]

    O_1s = BasisFunction([
        (exp, coeff, O_pos, (0, 0, 0))
        for exp, coeff in zip(O_S_exps, O_S_coeffs)
    ])
    basis_set[idx] = O_1s
    idx += 1

    # 2s and 2p (SP shell)
    O_SP_exps = [5.033151319, 1.169596125, 0.3803889600]
    O_SP_s_coeffs = [-0.09996722919, 0.3995128261, 0.7001154689]
    O_SP_p_coeffs = [0.1559162750, 0.6076837186, 0.3919573931]

    O_2s = BasisFunction([
        (exp, coeff, O_pos, (0, 0, 0))
        for exp, coeff in zip(O_SP_exps, O_SP_s_coeffs)
    ])
    basis_set[idx] = O_2s
    idx += 1

    # px, py, pz — same exps, p_coeffs
    for angmom in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
        func = BasisFunction([
            (exp, coeff, O_pos, angmom)
            for exp, coeff in zip(O_SP_exps, O_SP_p_coeffs)
        ])
        basis_set[idx] = func
        idx += 1

    # === Hydrogens ===
    H1_pos = [0.0, -1.43255657, 1.03903350] 
    H2_pos = [0.0,  1.43255657, 1.03903350]

    H_exps = [3.42525091, 0.62391373, 0.16885540]
    H_coeffs = [0.15432897, 0.53532814, 0.44463454]

    for H_pos in [H1_pos, H2_pos]:
        H_1s = BasisFunction([
            (exp, coeff, H_pos, (0, 0, 0))
            for exp, coeff in zip(H_exps, H_coeffs)
        ])
        basis_set[idx] = H_1s
        idx += 1

    return basis_set, [8.0, 1.0, 1.0], [O_pos, H1_pos, H2_pos]

basis_set, Z_nuc, R_nuc = build_H2O_sto3g_basis()

# Nuclear geometry
nuclei = list(zip(Z_nuc, R_nuc))

start_time = time.time()
scf_rhf(n_elec=10, R_nuc=R_nuc, Z_nuc=Z_nuc, verbose=1, basis_set=basis_set, nuclei=nuclei)
end_time = time.time()

print(f"Total runtime: {end_time - start_time:.4f} seconds")
