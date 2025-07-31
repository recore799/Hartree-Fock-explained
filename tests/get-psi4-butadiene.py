import psi4

psi4.set_options({
    "basis": "sto-3g",
    "scf_type": "pk",      # Pure Python backend
    "puream": False,       # Use Cartesian functions
    "reference": "rhf",    # Restricted HF
    "print": 1
})

# 1,3-Butadiene (C4H6) planar geometry in Bohr (approx from B3LYP/6-31G*)
# Geometry is symmetric and lies in the xy-plane
butadiene_geom_bohr = """
units bohr
C      0.000000     0.000000     0.000000
C      2.511087     0.000000     0.000000
C      4.674429     0.000000     0.000000
C      7.185516     0.000000     0.000000
H     -1.019300     0.000000     0.000000
H      2.511087    -1.936272     0.000000
H      2.511087     1.936272     0.000000
H      4.674429    -1.936272     0.000000
H      4.674429     1.936272     0.000000
H      8.204816     0.000000     0.000000
"""

mol = psi4.geometry(butadiene_geom_bohr)

# Compute nuclear repulsion energy
E_nuc_psi4 = mol.nuclear_repulsion_energy()
print(f"\nPsi4 nuclear repulsion (Bohr units): {E_nuc_psi4:.8f} Ha")

# Run SCF and compare total energy
E_scf = psi4.energy("scf")
print(f"Total SCF energy (Bohr units): {E_scf:.8f} Ha")
