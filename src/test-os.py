import numpy as np
from OS import compute_eri_tensor_sparse

# STO-3G parameters for Hydrogen (ζ=1.24)
slater_exponent = 1.24
H_STO3G_params = [
    (0.109818, 0.444635),  # (unscaled exponent, coefficient)
    (0.405771, 0.535328),
    (2.227660, 0.154329)
]

# Scale exponents by ζ^2 (1.24^2 = 1.5376)
H_STO3G_primitives = [
    (zeta_unscaled * slater_exponent**2, coeff)
    for zeta_unscaled, coeff in H_STO3G_params
]

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

H1 = BasisFunction([
    (z, c, [0.0, 0.0, 0.0], (0,0,0)) for z, c in H_STO3G_primitives
])
H2 = BasisFunction([
    (z, c, [1.4, 0.0, 0.0], (0,0,0)) for z, c in H_STO3G_primitives
])

basis_set = {0: H1, 1: H2}

# Compute (11|11) ERI
eri = compute_eri_tensor_sparse(basis_set)

print(eri)

# expected_eri_0000 = 0.7746  # Szabo/Ostlund value for H2 in STO-3G
# print(f"Computed (00|00): {eri_0000:.4f}")
# print(f"Expected (00|00): {expected_eri_0000:.4f}")
# print(f"Error: {abs(eri_0000 - expected_eri_0000):.4e}")
