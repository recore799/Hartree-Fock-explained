import psi4
import numpy as np

# Set up LiH molecule (R = 1.6 bohr)
psi4.geometry("""
Li 0 0 0
H  0 0 1.6
unit bohr
""")

# Configure calculation (STO-3G, disable spherical harmonics)
psi4.set_options({
    "basis": "sto-3g",
    "puream": False,  # Use Cartesian basis (required for minimal STO-3G)
    "scf_type": "pk",  # Conventional integral evaluation
})

# Compute core Hamiltonian (T + V_ne)
mints = psi4.core.MintsHelper(psi4.core.Wavefunction.build(psi4.core.get_active_molecule(), psi4.core.get_global_option("BASIS")))
T = mints.ao_kinetic().np  # Kinetic energy matrix
V = mints.ao_potential().np  # Nuclear attraction matrix
H_core = T + V

# Print results (3x3 matrix for Li 1s, 2s, H 1s)
print("Overlap Matrix (S):")
print(mints.ao_overlap().np[:3, :3])  # First 3 basis functions

print("\nCore Hamiltonian):")
print(H_core[:3, :3])

E = psi4.energy("scf")  # Automatically uses RHF
print("Total Energy (HeH+):", E)

