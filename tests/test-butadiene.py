import numpy as np
import time

from src.scf.rhf_os import scf_rhf  # Your engine

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

def build_butadiene_sto3g_basis():
    basis_set = {}
    idx = 0

    # === STO-3G Parameters ===
    # Carbon
    C_S_exps = [71.61683735, 13.04509632, 3.530512160]
    C_S_coeffs = [0.1543289673, 0.5353281423, 0.4446345422]

    C_SP_exps = [2.941249355, 0.6834830964, 0.2222899159]
    C_SP_s_coeffs = [-0.09996722919, 0.3995128261, 0.7001154689]
    C_SP_p_coeffs = [0.1559162750, 0.6076837186, 0.3919573931]

    # Hydrogen
    H_exps = [3.42525091, 0.62391373, 0.16885540]
    H_coeffs = [0.15432897, 0.53532814, 0.44463454]

    # === Geometry: Butadiene (C4H6), planar, Bohr units ===
    geom = [
        ("C", [ 0.000000,     0.000000,     0.000000 ]),
        ("C", [ 2.511087,     0.000000,     0.000000 ]),
        ("C", [ 4.674429,     0.000000,     0.000000 ]),
        ("C", [ 7.185516,     0.000000,     0.000000 ]),
        ("H", [-1.019300,     0.000000,     0.000000 ]),
        ("H", [ 2.511087,    -1.936272,     0.000000 ]),
        ("H", [ 2.511087,     1.936272,     0.000000 ]),
        ("H", [ 4.674429,    -1.936272,     0.000000 ]),
        ("H", [ 4.674429,     1.936272,     0.000000 ]),
        ("H", [ 8.204816,     0.000000,     0.000000 ]),
    ]

    Z_nuc = []
    R_nuc = []

    for sym, pos in geom:
        if sym == "C":
            Z_nuc.append(6.0)
        elif sym == "H":
            Z_nuc.append(1.0)
        R_nuc.append(pos)

    for sym, pos in geom:
        if sym == "C":
            # 1s contracted
            basis_set[idx] = BasisFunction([
                (exp, coeff, pos, (0, 0, 0))
                for exp, coeff in zip(C_S_exps, C_S_coeffs)
            ])
            idx += 1

            # 2s
            basis_set[idx] = BasisFunction([
                (exp, coeff, pos, (0, 0, 0))
                for exp, coeff in zip(C_SP_exps, C_SP_s_coeffs)
            ])
            idx += 1

            # 2p (x, y, z)
            for angmom in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
                basis_set[idx] = BasisFunction([
                    (exp, coeff, pos, angmom)
                    for exp, coeff in zip(C_SP_exps, C_SP_p_coeffs)
                ])
                idx += 1

        elif sym == "H":
            basis_set[idx] = BasisFunction([
                (exp, coeff, pos, (0, 0, 0))
                for exp, coeff in zip(H_exps, H_coeffs)
            ])
            idx += 1

    return basis_set, Z_nuc, R_nuc

# === Run RHF SCF ===
basis_set, Z_nuc, R_nuc = build_butadiene_sto3g_basis()
nuclei = list(zip(Z_nuc, R_nuc))

start_time = time.time()
scf_rhf(n_elec=30, R_nuc=R_nuc, Z_nuc=Z_nuc, verbose=1, basis_set=basis_set, nuclei=nuclei)
end_time = time.time()

print(f"Total runtime: {end_time - start_time:.4f} seconds")
