import psi4

psi4.set_options({
    "basis": "sto-3g",
    "scf_type": "pk",
    "puream": False,    # Cartesian basis
    "reference": "rhf",
    "print": 1
})

# Define CH4 geometry in Bohr (tetrahedral)
# Bond length ~1.09 Å = 2.059 Bohr
ch4_geom_bohr = """
units bohr
C   0.00000000   0.00000000   0.00000000
H   2.05900000   2.05900000   2.05900000
H  -2.05900000  -2.05900000   2.05900000
H  -2.05900000   2.05900000  -2.05900000
H   2.05900000  -2.05900000  -2.05900000
"""

mol = psi4.geometry(ch4_geom_bohr)

# Compute nuclear repulsion energy
E_nuc_psi4 = mol.nuclear_repulsion_energy()
print(f"\nPsi4 nuclear repulsion (Bohr units): {E_nuc_psi4:.8f} Ha")

# Run SCF and compare total energy
E_scf = psi4.energy("scf")
print(f"Total SCF energy (Bohr units): {E_scf:.8f} Ha")
