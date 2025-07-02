import psi4

# Set up H2O molecule
h2o = psi4.geometry("""
O
H 1 1.8
H 1 1.8 2 104.5
""")

# Set basis (STO-3G)
psi4.set_options({"basis": "sto-3g"})

# Compute ERIs
mints = psi4.core.MintsHelper(psi4.core.Wavefunction.build(h2o, psi4.core.get_global_option("basis")))
eri = mints.ao_eri().np  # Get all ERIs in AO basis

# Extract (O1s O1s | H1s H1s)
print("PSI4 (O1s O1s | H1s H1s):", eri[0, 0, 3, 3])  # Indices depend on basis ordering!
