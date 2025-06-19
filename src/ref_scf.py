import numpy as np
from sto3g_basis import build_sto3g_basis_2s
from integrals import compute_eri_tensor_sparse
from pyscf import gto, scf, ao2mo

# LiH calculation (STO-3G basis, R=3.014 bohr ~ 1.6 Å)
mol = gto.M(
    atom = 'Li 0 0 0; H 0 0 3.014',
    basis = 'sto3g',
)
mf = scf.RHF(mol).run()

# Get ERI tensor in atomic orbitals (chemists' notation: (μν|λσ))
eri_pyscf = ao2mo.restore(1, mf._eri, mol.nao)  # Shape: (nbf, nbf, nbf, nbf)

# Print key integrals for comparison
print("PySCF ERI tensor (STO-3G LiH):")
print("(Li 1s Li 1s | Li 1s Li 1s):", eri_pyscf[0,0,0,0])  # ~1.8-2.0
print("(Li 1s H 1s | Li 1s H 1s):", eri_pyscf[0,2,0,2])    # ~0.6-0.8
print("(H 1s H 1s | H 1s H 1s):", eri_pyscf[2,2,2,2])      # ~0.7-0.9



primitives_lithium = [
    build_sto3g_basis_2s(6.3, shell="1s"),
    build_sto3g_basis_2s(1.3, shell="2s"),
]

primitives_hydrogen = [
    build_sto3g_basis_2s(1.0, shell="1s"),
]

primitives_lih = primitives_lithium + primitives_hydrogen

pos_lih = np.array([[0,0,0],[0,0,0],[0,0,3.014]])

your_eri = compute_eri_tensor_sparse(primitives_lih, pos_lih)

# Key comparisons
print("\nYour ERI tensor:")
print("(Li 1s Li 1s | Li 1s Li 1s):", your_eri[0,0,0,0])
print("(Li 1s H 1s | Li 1s H 1s):", your_eri[0,2,0,2])
print("(H 1s H 1s | H 1s H 1s):", your_eri[2,2,2,2])

# Symmetry check (should be True)
print("\nSymmetry checks:")
print("(μν|λσ) == (νμ|λσ):", np.allclose(your_eri, np.transpose(your_eri, (1,0,2,3))))
print("(μν|λσ) == (λσ|μν):", np.allclose(your_eri, np.transpose(your_eri, (2,3,0,1))))


