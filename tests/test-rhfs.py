import numpy as np
from src.scf.rhf_s import scf_rhf
from src.scf.sto3g_basis import build_sto3g_basis

# Quick test for H2 and HeH+
print("Testing SCF implementation...")
print("\n1. H2 molecule at R = 1.4 au")
print("-" * 60)

# For H2 (H: Zeta=1.24)
sto3g_h = build_sto3g_basis(zeta=1.24)    # Hydrogen basis

primitives_h2 = [sto3g_h, sto3g_h]  # [He, H]


pos_h2 = np.array([[0,0,0],[1.4,0,0]])
Z_h2 = (1.0,1.0)
scf_rhf(primitives_h2, pos=pos_h2, R=1.4, Z=Z_h2, n_elec=2, R_nuc=pos_h2, Z_nuc=Z_h2, verbose=1)

print("\n2. HeH+ ion at R = 1.4 au")
print("-" * 60)

# For HeH+ (He: Zeta=2.095, H: Zeta=1.24)
sto3g_he = build_sto3g_basis(zeta=2.095)  # Helium basis

primitives_heh = [sto3g_he, sto3g_h]  # [He, H]

pos_heh = np.array([[0,0,0],[1.4632,0,0]])
Z_heh = (2.0,1.0)
scf_rhf(primitives_heh, pos=pos_heh, R=1.4632, Z=Z_heh, n_elec=2, R_nuc=pos_heh, Z_nuc=Z_heh, verbose=1)

