import psi4
import numpy as np

# Set up H2O molecule (experimental geometry)
h2o = psi4.geometry("""
O
H 1 0.96
H 1 0.96 2 104.5
symmetry c1  # Disable symmetry to avoid irreps issues
""")

# Set options
psi4.set_options({
    'basis': 'sto-3g',  # STO-3G basis set
    'scf_type': 'pk',   # Conventional SCF (not density-fitted)
    'print': 1          # Low verbosity
})

# Run SCF and get wavefunction
scf_energy, wfn = psi4.energy('scf', return_wfn=True)

# Get matrices
mints = psi4.core.MintsHelper(wfn.basisset())

# Overlap matrix (S)
S = np.array(mints.ao_overlap())
print("\nOverlap Matrix (S) shape:", S.shape)
print(S)

# Core Hamiltonian (H) = Kinetic + Nuclear Attraction
T = np.array(mints.ao_kinetic())  # Kinetic energy
V = np.array(mints.ao_potential()) # Nuclear attraction
H = T + V
print("\nCore Hamiltonian (H) shape:", H.shape)
print(H)

# Density matrix (P) from occupied orbitals
C = wfn.Ca().to_array()  # MO coefficients
occ = wfn.nalpha()       # Number of alpha electrons
P = np.dot(C[:, :occ], C[:, :occ].T)
print("\nDensity Matrix (P) shape:", P.shape)
print(P)

# Fock matrix (F) = H + J - K
F = wfn.Fa().to_array()  # Alpha Fock matrix
print("\nFock Matrix (F) shape:", F.shape)
print(F)

# Two-electron integrals (ERI) in AO basis
ERI = np.array(mints.ao_eri())
print("\nERI Tensor shape:", ERI.shape)

# Save matrices
# np.savetxt("S_H2O.txt", S)
# np.savetxt("F_H2O.txt", F)
# np.save("ERI_H2O.npy", ERI)  # 4D tensor (binary format)

# Save matrices with human-readable formatting
def save_human_readable(filename, array, precision=8):
    """Save a NumPy array in a human-readable text format."""
    with open(filename, 'w') as f:
        if array.ndim <= 2:
            np.savetxt(f, array, fmt=f'%.{precision}f')
        elif array.ndim == 4:  # ERI tensor
            for i in range(array.shape[0]):
                for j in range(array.shape[1]):
                    f.write(f"ERI Block (i={i}, j={j}):\n")
                    np.savetxt(f, array[i, j], fmt=f'%.{precision}f')
                    f.write("\n")

# Save all matrices
save_human_readable("S_H2O_hr.txt", S)          # Overlap (shape: nao × nao)
save_human_readable("F_H2O_hr.txt", F)          # Fock (shape: nao × nao)
save_human_readable("ERI_H2O_hr.txt", ERI)      # ERI tensor (shape: nao × nao × nao × nao)

# Optional: Save metadata (basis set, geometry, energy)
with open("metadata_H2O.txt", 'w') as f:
    f.write(f"SCF Energy (Hartree): {scf_energy}\n")
    f.write(f"Basis Set: {wfn.basisset().name()}\n")
    f.write("Geometry (Å):\n")
    f.write(h2o.save_string_xyz())

print("\nSCF Energy (Hartree):", scf_energy)
