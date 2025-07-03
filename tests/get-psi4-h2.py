import psi4
import numpy as np

# Set up H2 molecule (bond length = 0.74 Angstrom)
h2 = psi4.geometry("""
H
H 1 0.74
symmetry c1  # Disable symmetry to avoid irreps issues
""")

# Set options
psi4.set_options({
    'basis': 'sto-3g',
    'scf_type': 'pk',
    'print': 1
})

# Run SCF and get wavefunction
scf_energy, wfn = psi4.energy('scf', return_wfn=True)

# Get matrices
mints = psi4.core.MintsHelper(wfn.basisset())

# Overlap matrix (S)
S = np.array(mints.ao_overlap())
print("\nOverlap Matrix (S):\n", S)

# Core Hamiltonian (H) = Kinetic + Nuclear Attraction
H = np.array(mints.ao_kinetic()) + np.array(mints.ao_potential())
print("\nCore Hamiltonian (H):\n", H)

# Density matrix (P) from occupied orbitals
C = wfn.Ca().to_array()  # Convert Psi4 Matrix to NumPy array properly
occ = wfn.nalpha()       # Number of alpha electrons
P = np.dot(C[:, :occ], C[:, :occ].T)
print("\nDensity Matrix (P):\n", P)

# Fock matrix (F) = H + J - K
F = wfn.Fa().to_array()  # Convert Psi4 Matrix to NumPy array
print("\nFock Matrix (F):\n", F)

# Two-electron integrals (ERI) in AO basis
ERI = np.array(mints.ao_eri())
print("\nERI Tensor (shape=%s):" % str(ERI.shape))
print(ERI)

# Save matrices
np.savetxt("S.txt", S)
np.savetxt("F.txt", F)
np.save("ERI.npy", ERI)
