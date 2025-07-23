import numpy as np
from src.scf.rhf_os import scf_rhf
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

def build_LiH_sto3g_basis():
    basis_set = {}
    idx = 0

    # === Li ===
    Li_pos = [0.0, 0.0, 0.0]

    # STO-3G for Li
    Li_S_exps = [16.11957475, 2.936200663, 0.7946504870]
    Li_S_coeffs = [0.1543289673, 0.5353281423, 0.4446345422]

    Li_SP_exps = [0.6362897469, 0.1478600533, 0.04808867840]
    Li_SP_s_coeffs = [-0.09996722919, 0.3995128261, 0.7001154689]
    Li_SP_p_coeffs = [0.1559162750, 0.6076837186, 0.3919573931]

    # Li 1s
    Li_1s = BasisFunction([
        (exp, coeff, Li_pos, (0, 0, 0))
        for exp, coeff in zip(Li_S_exps, Li_S_coeffs)
    ])
    basis_set[idx] = Li_1s
    idx += 1

    # Li 2s
    Li_2s = BasisFunction([
        (exp, coeff, Li_pos, (0, 0, 0))
        for exp, coeff in zip(Li_SP_exps, Li_SP_s_coeffs)
    ])
    basis_set[idx] = Li_2s
    idx += 1

    # Li 2p (x, y, z)
    for angmom in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
        func = BasisFunction([
            (exp, coeff, Li_pos, angmom)
            for exp, coeff in zip(Li_SP_exps, Li_SP_p_coeffs)
        ])
        basis_set[idx] = func
        idx += 1

    # === Hydrogen ===
    H_pos = [0.0, 0.0, 2.64562]

    H_S_exps = [3.42525091, 0.62391373, 0.16885540]
    H_S_coeffs = [0.15432897, 0.53532814, 0.44463454]

    H_1s = BasisFunction([
        (exp, coeff, H_pos, (0, 0, 0))
        for exp, coeff in zip(H_S_exps, H_S_coeffs)
    ])
    basis_set[idx] = H_1s
    idx += 1

    return basis_set, [3.0, 1.0], [Li_pos, H_pos]  # Z_nuc, R_nuc

# ==== Run SCF ====
basis_set, Z_nuc, R_nuc = build_LiH_sto3g_basis()
nuclei = list(zip(Z_nuc, R_nuc))

start_time = time.time()
scf_rhf(n_elec=4, R_nuc=R_nuc, Z_nuc=Z_nuc, verbose=1, basis_set=basis_set, nuclei=nuclei)
end_time = time.time()

print(f"Total runtime: {end_time - start_time:.4f} seconds")
