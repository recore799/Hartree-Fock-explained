import numpy as np
from OS import compute_eri_element, boys_sequence

# STO-3G parameters for Hydrogen (ζ=1.24)
slater_exponent = 1.24
H_STO3G_params = [
    (0.109818, 0.444635),
    (0.405771, 0.535328),
    (2.227660, 0.154329)
]

# Scale exponents by ζ^2 (1.24^2 = 1.5376)
H_STO3G_primitives = [
    (zeta_unscaled * slater_exponent**2, coeff)
    for zeta_unscaled, coeff in H_STO3G_params
]

# Artificial p-orbital parameters (using standard 6-31G exponents for testing)
P_EXPONENTS = [0.8, 0.2]  # Two p-type primitives
P_COEFFS = [0.6, 0.4]      # Corresponding coefficients

class Primitive:
    def __init__(self, zeta, coeff, center, angmom=(0,0,0)):
        self.zeta = zeta
        self.coeff = coeff
        self.center = np.array(center)
        self.angmom = angmom

class BasisFunction:
    def __init__(self, primitives):
        self.primitives = primitives

# Test parameters
R = 1.4  # H-H distance in bohr
zeta_s = 1.24  # Slater exponent for H

# Simple s-type Gaussian (for testing)
s_orbital = BasisFunction([
    Primitive(zeta_s, 1.0, [0.0, 0.0, 0.0], (0,0,0))
])

# Simple p-type Gaussian (artificial exponents for testing)
px_orbital = BasisFunction([
    Primitive(0.1, 1.0, [0.0, 0.0, 0.0], (1,0,0)),
])

# Create basis set
basis_set = {
    0: s_orbital,                           # s at origin
    1: px_orbital,                          # px at origin
    2: BasisFunction([Primitive(zeta_s, 1.0, [R, 0.0, 0.0], (0,0,0))]),  # s at (R,0,0)
    3: BasisFunction([Primitive(0.8, 0.6, [R, 0.0, 0.0], (1,0,0))])      # px at (R,0,0)
}

# ===== Analytical Test Values =====
from scipy.special import hyp1f1
def boys_exact(n, T):
    return 0.5 * T**(-n - 0.5) * hyp1f1(n + 0.5, n + 1.5, -T)

def analytical_ssss(zeta):
    """(ss|ss) for same-center s orbitals"""
    return (5/8)*(np.pi/zeta)**0.5

def analytical_pxpx_pxpx(zeta):
    """(px px|px px) for same-center p orbitals"""
    return (15/16)*(np.pi/zeta)**0.5

def analytical_ss_ss(R, zeta):
    """(s_A s_A | s_B s_B) between two atoms"""
    T = zeta * R**2
    F = boys_sequence(0, T)
    return 2 * np.pi**(5/2) / (zeta**2 * np.sqrt(2*zeta)) * F[0]

# ===== Test Cases =====
print("\n=== Same Center Tests ===")
# (ss|ss)
eri_ssss = compute_eri_element(0, 0, 0, 0, basis_set)
expected = analytical_ssss(zeta_s)
print(f"(ss|ss): {eri_ssss:.6f} (expected {expected:.6f})")
# assert np.isclose(eri_ssss, expected, rtol=1e-4)

# (px px|px px)
eri_pxpx = compute_eri_element(1, 1, 1, 1, basis_set)
expected = analytical_pxpx_pxpx(1.0)  # Using dominant exponent
print(f"(px px|px px): {eri_pxpx:.6f} (expected {expected:.6f})")
# assert np.isclose(eri_pxpx, expected, rtol=1e-2)

print("\n=== Two-Center Tests ===")
# (s_A s_A | s_B s_B)
eri_ss_ss = compute_eri_element(0, 0, 2, 2, basis_set)
expected = analytical_ss_ss(R, zeta_s)
print(f"(s s|s s) R=1.4: {eri_ss_ss:.6f} (expected {expected:.6f})")
# assert np.isclose(eri_ss_ss, expected, rtol=1e-4)

# (s_A px_A | s_B px_B)
eri_spx_spx = compute_eri_element(0, 1, 2, 3, basis_set)
print(f"(s px|s px) R=1.4: {eri_spx_spx:.6f} (no simple analytical)")

# (s_A s_A | px_B px_B)
eri_ss_pxpx = compute_eri_element(0, 0, 3, 3, basis_set)
print(f"(s s|px px) R=1.4: {eri_ss_pxpx:.6f}")

