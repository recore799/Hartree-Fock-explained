import numpy as np
from OS import ObaraSaikaERI

# STO-3G basis coefficients for O and H (example values)
# Format: (exponent, contraction coefficient)
O_1s = [(5.098112, 0.154329), (0.920811, 0.535328), (0.283205, 0.444635)]
O_2s = [(0.920811, -0.099967), (0.283205, 0.399513)]
O_2p = [(0.920811, 0.155916), (0.283205, 0.607684)]
H_1s = [(0.168856, 0.444635), (0.623913, 0.535328), (3.42525, 0.154329)]


# H2O geometry (atomic units, Bohr)
O_pos = np.array([0.0, 0.0, 0.0])
H1_pos = np.array([0.0, 1.432, -1.107])  # ~1.8 Å OH bond
H2_pos = np.array([0.0, -1.432, -1.107])


# Example: (O1s O1s | H1s H1s) - all primitives
eri_calculator = ObaraSaikaERI()

# Take the first primitive of O 1s and H 1s
zeta_O1s, coeff_O1s = O_1s[0]
zeta_H1s, coeff_H1s = H_1s[0]

# Compute ERI
eri_primitive = eri_calculator.compute_eri(
    zeta_O1s, zeta_O1s, zeta_H1s, zeta_H1s,
    O_pos, O_pos, H1_pos, H1_pos,
    (0,0,0), (0,0,0), (0,0,0), (0,0,0)
)
print(f"(O1s O1s | H1s H1s) primitive ERI: {eri_primitive:.8f}")
