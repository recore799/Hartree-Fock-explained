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

def build_H2_sto3g_basis():
    basis_set = {}
    idx = 0

    # === Hydrogen ===
    H_pos1 = [0.0, 0.0, 0.0]
    H_pos2 = [0.0, 0.0, 1.4]

    H_S_exps = [3.42525091, 0.62391373, 0.16885540]
    H_S_coeffs = [0.15432897, 0.53532814, 0.44463454]

    H_1s1 = BasisFunction([
        (exp, coeff, H_pos1, (0, 0, 0))
        for exp, coeff in zip(H_S_exps, H_S_coeffs)
    ])
    basis_set[idx] = H_1s1
    idx += 1
    H_1s2 = BasisFunction([
        (exp, coeff, H_pos2, (0, 0, 0))
        for exp, coeff in zip(H_S_exps, H_S_coeffs)
    ])
    basis_set[idx] = H_1s2

    return basis_set, [1.0, 1.0], [H_pos1, H_pos2]  # Z_nuc, R_nuc

# ==== Run SCF ====
basis_set, Z_nuc, R_nuc = build_H2_sto3g_basis()
nuclei = list(zip(Z_nuc, R_nuc))

start_time = time.time()
scf_rhf(n_elec=2, R_nuc=R_nuc, Z_nuc=Z_nuc, verbose=1, basis_set=basis_set, nuclei=nuclei)
end_time = time.time()

print(f"Total runtime: {end_time - start_time:.4f} seconds")
