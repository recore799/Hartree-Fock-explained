import psi4

psi4.set_options({
    "basis": "sto-3g",
    "scf_type": "pk",
    "puream": False,    # Cartesian basis
    "reference": "rhf",
    "print": 1
})

# Define H2O geometry in Bohr (matches your custom code)
h2o_geom_bohr = """
units bohr
O  0.0 0.0 0.0
H  0.0 -1.43255657 1.03903350
H  0.0  1.43255657 1.03903350
"""

mol = psi4.geometry(h2o_geom_bohr)

# Compute nuclear repulsion energy
E_nuc_psi4 = mol.nuclear_repulsion_energy()
print(f"\nPsi4 nuclear repulsion (Bohr units): {E_nuc_psi4:.8f} Ha")

# Run SCF and compare total energy
E_scf = psi4.energy("scf")
print(f"Total SCF energy (Bohr units): {E_scf:.8f} Ha")
