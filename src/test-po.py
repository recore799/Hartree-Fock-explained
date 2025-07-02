import numpy as np
from OS import compute_eri_element

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

# Two centers separated in x-direction
center_A = [0.0, 0.0, 0.0]
center_B = [1.0, 0.0, 0.0]

primitive_params =  H_STO3G_params # arbitrary but reasonable

px_A = BasisFunction([
    (primitive_params[0][0], primitive_params[0][1], center_A, (1, 0, 0)),
    (primitive_params[1][0], primitive_params[1][1], center_A, (1, 0, 0)),
    (primitive_params[2][0], primitive_params[2][1], center_A, (1, 0, 0)),
])


px_B = BasisFunction([
    (primitive_params[0][0], primitive_params[0][1], center_A, (1, 0, 0)),
    (primitive_params[1][0], primitive_params[1][1], center_A, (1, 0, 0)),
    (primitive_params[2][0], primitive_params[2][1], center_A, (1, 0, 0)),
])



basis_set = {
    0: px_A,
    1: px_B,
}


eri_pxpx = compute_eri_element(0, 0, 1, 1, basis_set)

print("(px px | px px) =", eri_pxpx)
