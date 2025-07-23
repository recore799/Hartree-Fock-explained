import psi4
import numpy as np

# Define molecule and SCF options
psi4.set_options({
    "basis": "sto-3g",
    "scf_type": "pk",
    "puream": False,
    "reference": "rhf",
    "guess": "core",
    "print": 3
})

# Define molecule
mol = psi4.geometry("""
units bohr
0 1
Li 0.0 0.0 0.0
H  0.0 0.0 2.64562
""")

# Run SCF and grab the wavefunction
E, wfn = psi4.energy("scf", return_wfn=True)

# Grab MintsHelper for integral access
mints = psi4.core.MintsHelper(wfn.basisset())

# Get matrices and convert to NumPy
S = np.array(mints.ao_overlap())
T = np.array(mints.ao_kinetic())
V = np.array(mints.ao_potential())
H = T + V
ERI = np.array(mints.ao_eri())

# Formatting options
np.set_printoptions(
    precision=6,  # 6 decimal places
    suppress=True, # Suppress scientific notation
    linewidth=100  # Line width for printing
)

# Print formatted matrices
def print_matrix(name, matrix):
    print(f"\n{name} ({matrix.shape}):")
    print(matrix)

print_matrix("Overlap (S)", S)
print_matrix("Kinetic (T)", T)
print_matrix("Nuclear Attraction (V)", V)
print_matrix("Core Hamiltonian (H)", H)

# For ERI, print a representative slice
print("\nERI tensor (4D) - First 2x2x2x2 block:")
print(ERI[:2,:2,:2,:2])

# Print full ERI with indices (optional)
# print("\nFull ERI tensor with indices:")
# nbf = S.shape[0]
# for i in range(nbf):
#     for j in range(nbf):
#         for k in range(nbf):
#             for l in range(nbf):
#                 val = ERI[i,j,k,l]
#                 if abs(val) > 1e-6:  # Only print significant values
#                     print(f"({i+1}{j+1}|{k+1}{l+1}) = {val:.6f}")
