import psi4
import numpy as np

# Set up H2 with p-functions
psi4.geometry("""
H 0 0 0
H 1.4 0 0
""")

psi4.set_options({
    'basis': 'sto-3g',
    'scf_type': 'pk',
    'puream': False  # Use Cartesian (px,py,pz) basis
})

# Compute ERIs
psi4.energy('scf')
wfn = psi4.core.Wavefunction.build(molecule, psi4.core.get_global_option('basis'))
mints = psi4.core.MintsHelper(wfn.basisset())
eri_psi4 = np.array(mints.ao_eri())

# Get basis function ordering
basis_info = []
for i in range(wfn.nso()):
    center = wfn.molecule().xyz(wfn.basisset().function_to_center(i))
    l = wfn.basisset().shell(wfn.basisset().function_to_shell(i)).am
    basis_info.append((i, center, l))

print("Basis function ordering:")
for info in basis_info:
    print(info)
