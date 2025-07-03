import numpy as np
from scipy.linalg import eigh, fractional_matrix_power
from integrals import build_integral_arrays, canonical_eri_key
from sto3g_basis import build_sto3g_basis, build_sto3g_basis_2s
from OS import compute_eri_tensor_sparse, compute_overlap_matrix, compute_hcore

def scf_rhf(
        # primitives: list[list[tuple[float, float]]],
        # pos: np.ndarray, R: float, Z: tuple, n_elec: int,
        n_elec, 
        R_nuc: np.ndarray, Z_nuc: list, basis_set, nuclei,
        max_iter=50, conv_tol=1e-6, verbose=1
        ) -> dict:
    """
    Restricted Hartree-Fock SCF calculation.
    Args:
        primitives: list of two lists containing (alpha, coeff) pairs for each atom
        pos: np.ndarray of 3D nuclear positions
        R: bond distance in atomic units
        Z: nuclear charges of each atom
        max_iter: maximum number of SCF iterations
        conv_tol: convergence tolerance for density matrix
        verbose: whether to print iteration details
    
    Returns:
        dict containing final energy, orbitals, density matrix, etc.
    """
    
    # Build integral arrays
    # S_mat, H, test = build_integral_arrays(primitives, pos, Z)
    eri_dict = compute_eri_tensor_sparse(basis_set)
    S = compute_overlap_matrix(basis_set)
    T, V, H_core = compute_hcore(basis_set, nuclei)

   
    # Number of basis functions and electrons
    nbf = S.shape[0]
    n_occ = n_elec // 2  # Number of occupied orbitals (doubly occupied)
    
    if verbose >= 1:
        print(f"Number of basis functions: {nbf}")
        print(f"Number of electrons: {n_elec}")
        print(f"Number of occupied orbitals: {n_occ}")
        # print(f"Bond distance: {R:.3f} au")
        # print("-" * 50)
    
    # Symmetric orthogonalization: X = S^(-1/2)
    # This transforms the basis to an orthonormal one
    try:
        X = fractional_matrix_power(S, -0.5)
    except np.linalg.LinAlgError:
        # If S^(-1/2) is too small, use canonical orthogonalization
        eigvals, eigvecs = eigh(S)
        X = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T


    # Initial guess: core Hamiltonian
    P = np.zeros_like(S)
    E_old = 0.0
    
    # SCF iterations
    for iteration in range(max_iter):
        
        # Build Fock matrix
        F = build_fock_matrix_sparse_OS(H_core, P, eri_dict)
        
        # Calculate electronic energy
        E_elec = np.sum(P * H_core) + 0.5 * np.sum(P * (F - H_core)) # More numerically stable       
        # Nuclear repulsion energy (generalized for polyatomics)
        E_nuc = compute_nuclear_repulsion_energy(Z_nuc, R_nuc)

        E_total = E_elec + E_nuc
        
        if verbose >= 3:
            print(f"Iteration {iteration + 1:2d}: E_elec = {E_elec:12.8f}, "
                  f"E_total = {E_total:12.8f}")
        
        # Transform Fock matrix to orthogonal basis: F' = X^T * F * X
        F_prime = X.T @ F @ X
        
        # Diagonalize F' to get orbital energies and coefficients
        orbital_energies, C_prime = eigh(F_prime)
        
        # Transform back to original (non-orthogonal) basis: C = X * C'
        C = X @ C_prime
        
        # Build new density matrix
        # P_μν = 2 * Σ_i^{occ} C_μi * C_νi
        P_new = 2.0 * C[:, :n_occ] @ C[:, :n_occ].T
        
        # Check convergence (RMS change in density matrix)
        delta_P = P_new - P
        rms_change = np.sqrt(np.mean(delta_P**2))
        
        if verbose >= 3:
            print(f"           RMS density change = {rms_change:.2e}")
        
        # Check for convergence
        if rms_change < conv_tol and iteration > 0:
            if verbose >= 3:
                print(f"\nSCF converged in {iteration + 1} iterations!")
                print(f"Final electronic energy: {E_elec:.8f} au")
                print(f"Final total energy:      {E_total:.8f} au")
            break
        
        P = P_new
        E_old = E_elec
        
        if verbose >= 3:
            print()
    
    else:
        print(f"SCF did not converge in {max_iter} iterations!")
    
    # Calculate final properties
    results = {
        'energy_electronic': E_elec,
        'energy_nuclear': E_nuc,
        'energy_total': E_total,
        'orbital_energies': orbital_energies,
        'orbital_coefficients': C,
        'density_matrix': P,
        'fock_matrix': F,
        'overlap_matrix': S,
        'core_hamiltonian': H_core,
        'iterations': iteration + 1,
        'converged': rms_change < conv_tol,
        'eri_tensor': eri_dict,
        'T' : T,
        'V' : V
    }
    
    if verbose >= 1:
        print_final_results(results)
    
    return



def get_initial_guess(S: np.ndarray, Z_list: list) -> np.ndarray:
    """Guess based on atomic charges"""
    P = np.zeros_like(S)
    n_elec_per_atom = [Z for Z in Z_list]  # Neutral atoms
    
    # Simple diagonal guess (Li: 1s²2s¹, H: 1s¹)
    P[0,0] = 2.0  # Li 1s
    P[1,1] = 1.0  # Li 2s (partial)
    P[2,2] = 1.0  # H 1s
    
    return P

def compute_nuclear_repulsion_energy(Z_nuc, R_nuc):
    E_nuc = 0.0
    for i in range(len(Z_nuc)):
        for j in range(i + 1, len(Z_nuc)):
            Rij = np.linalg.norm(np.array(R_nuc[i]) - np.array(R_nuc[j]))
            if Rij > 1e-12:
                E_nuc += Z_nuc[i] * Z_nuc[j] / Rij
    return E_nuc


def build_fock_matrix(H_core: np.ndarray, P: np.ndarray, eri: np.ndarray) -> np.ndarray:
    """
    Build the Fock matrix: F_μν = H_μν^core + Σ_λσ P_λσ [(μν|λσ) - 0.5*(μλ|νσ)]
    """
    nbf = H_core.shape[0]
    G = np.zeros_like(H_core)
    
    # Two-electron contribution
    for mu in range(nbf):
        for nu in range(nbf):
            for lam in range(nbf):
                for sig in range(nbf):
                    G[mu, nu] += P[lam, sig] * (eri[mu, nu, lam, sig] - 0.5 * eri[mu, lam, nu, sig])


    return H_core + G


def build_fock_matrix_sparse(H_core: np.ndarray, P: np.ndarray, eri_dict: dict) -> np.ndarray:
    """
    Build the Fock matrix: F_μν = H_μν^core + Σ_λσ P_λσ [(μν|λσ) - 0.5*(μλ|νσ)]
    """
    nbf = H_core.shape[0]
    G = np.zeros_like(H_core)

    for mu in range(nbf):
        for nu in range(nbf):
            for lam in range(nbf):
                for sig in range(nbf):
                    key1 = canonical_eri_key(mu, nu, lam, sig)  # (μν|λσ)
                    key2 = canonical_eri_key(mu, lam, nu, sig)  # (μλ|νσ)
                    eri1 = eri_dict[key1]
                    eri2 = eri_dict[key2]
                    G[mu, nu] += P[lam, sig] * (eri1 - 0.5 * eri2)

    return H_core + G

def pack_index(mu, nu):
    # Pack two indices into one unique key (assuming mu >= nu)
    return mu * (mu + 1) // 2 + nu

def get_canonical_key(mu, nu, lam, sig):
    # Exploit 8-fold ERI symmetry: (ij|kl) == (ji|kl) == ...
    munu = pack_index(max(mu, nu), min(mu, nu))
    lamsig = pack_index(max(lam, sig), min(lam, sig))
    # Order the two pairs to enforce (munu >= lamsig)
    return (munu, lamsig) if munu >= lamsig else (lamsig, munu)




def build_fock_matrix_sparse_OS(H_core: np.ndarray, P: np.ndarray, eri_dict: dict) -> np.ndarray:
    """
    F_μν = H_μν^core + Σ_λσ P_λσ [(μν|λσ) - 0.5*(μλ|νσ)]
    """
    nbf = H_core.shape[0]
    G = np.zeros_like(H_core)

    for mu in range(nbf):
        for nu in range(nbf):
            for lam in range(nbf):
                for sig in range(nbf):
                    key1 = get_canonical_key(mu, nu, lam, sig)
                    key2 = get_canonical_key(mu, lam, nu, sig)

                    eri1 = eri_dict.get(key1, 0.0)
                    eri2 = eri_dict.get(key2, 0.0)

                    G[mu, nu] += P[lam, sig] * (eri1 - 0.5 * eri2)

    return H_core + G


def print_final_results(results):
    """Print formatted final SCF results"""
    print("\n" + "="*60)
    print("FINAL SCF RESULTS")
    print("="*60)

    print(f"{'Electronic energy:':<25}{results['energy_electronic']:>12.6f}  Ha")
    print(f"{'Nuclear repulsion:':<25}{results['energy_nuclear']:>12.6f}  Ha")
    print(f"{'Total energy:':<25}{results['energy_total']:>12.6f}  Ha")
    print(f"{'SCF iterations:':<25}{results['iterations']:>12d}")
    print(f"{'Converged:':<25}{str(results['converged']):>12}")
    
    print("\nOrbital energies (Ha):")
    for i, energy in enumerate(results['orbital_energies']):
        print(f"  ε_{i+1} = {energy:>10.6f}")
    
    print("\nOrbital Coefficients (C):")
    C = results['orbital_coefficients']
    for i in range(C.shape[1]):
        coeffs = "  ".join(f"{val:>8.4f}" for val in C[:, i])
        print(f"  Orbital {i+1:<2}: {coeffs}")

    print("S matrix:\n", results['overlap_matrix'])
    print("T matrix:\n", results['T'])
    print("V matrix:\n", results['V'])
    print("P matrix:\n", results['density_matrix'])
    print("C matrix:\n", results['orbital_coefficients'])
    print("H matrix:\n", results['core_hamiltonian'])

    eri = results['eri_tensor']
    eri_size = len(eri)

    print("ERI tensor shape:\n", eri_size)
    # if 'eri_tensor' in results:
    #     print("\nUnique Electron Repulsion Integrals (ERIs):")
    #     eri_dict = results['eri_tensor']
    #     for key, value in eri_dict.items():
    #         munu, lamsig = key  # Unpack the canonical key
    #         print(f"({munu}|{lamsig}): {value:.6f}")

# Example usage and test cases
if __name__ == "__main__":
    # # STO-3G basis parameters (alpha, contraction_coefficient)
    # # These are the normalized STO-3G parameters
    
    print("Testing SCF implementation...")
    print("\n1. H2 molecule at R = 1.4 au")
    print("-" * 60)
 
    # For H2 (H: Zeta=1.24)
    sto3g_h = build_sto3g_basis(zeta=1.24)    # Hydrogen basis

    primitives_h2 = [sto3g_h, sto3g_h]  # [He, H]
    

    pos_h2 = np.array([[0,0,0],[1.4,0,0]])
    Z_h2 = (1.0,1.0)

    # STO-3G parameters for Hydrogen (ζ=1.24)
    slater_exponent = 1.24
    H_STO3G_params = [
        (0.109818, 0.444635),  # (unscaled exponent, coefficient)
        (0.405771, 0.535328),
        (2.227660, 0.154329)
    ]

    # Scale exponents by ζ^2 (1.24^2 = 1.5376)
    H_STO3G_primitives = [
        (zeta_unscaled * slater_exponent**2, coeff)
        for zeta_unscaled, coeff in H_STO3G_params
    ]

    class Primitive:
        def __init__(self, zeta, coeff, center, angmom):
            self.zeta = zeta
            self.coeff = coeff
            self.center = np.array(center)
            self.angmom = angmom

    class BasisFunction:
        def __init__(self, primitives):
            self.primitives = [Primitive(zeta, coeff, center, angmom)
                               for zeta, coeff, center, angmom in primitives]

    H1 = BasisFunction([
        (z, c, [0.0, 0.0, 0.0], (0,0,0)) for z, c in H_STO3G_primitives
    ])
    H2 = BasisFunction([
        (z, c, [1.4, 0.0, 0.0], (0,0,0)) for z, c in H_STO3G_primitives
    ])

    basis_set = {0: H1, 1: H2}
    nuclei = [
        (1.0, [0.0, 0.0, 0.0]),
        (1.0, [1.4, 0.0, 0.0])
    ]


    scf_rhf(primitives_h2, pos=pos_h2, R=1.4, Z=Z_h2, n_elec=2, R_nuc=pos_h2, Z_nuc=Z_h2, verbose=1, basis_set=basis_set, nuclei=nuclei)





    # print("\nTesting He2 at R = 5.67 Bohr")
    # print("-" * 60)

    # # For HeH+ (He: Zeta=2.095, H: Zeta=1.24)
    # sto3g_he = build_sto3g_basis(zeta=1.6875)  # Helium basis

    # primitives_heh = [sto3g_he, sto3g_he]  # [He, H]

    # # print("Primitives_heh:\n", primitives_heh)
    
    # pos_heh = np.array([[0,0,0],[5.67,0,0]])
    # Z_heh = (2.0,2.0)
    # # scf_rhf(primitives_heh, pos=pos_heh, R=5.67, Z=Z_heh, n_elec=4, R_nuc=pos_heh, Z_nuc=Z_heh, verbose=1)

    # print("Testing SCF implementation...")
    # print("\n3. LiH molecule at R = 1.6 au")
    # print("-" * 40)
 
    # # primitives_lithium = [
    # #     build_sto3g_basis_2s(6.3, shell="1s"),
    # #     build_sto3g_basis_2s(1.0, shell="2s"),
    # # ]

    # # primitives_hydrogen = [
    # #     build_sto3g_basis_2s(1.0, shell="1s"),
    # # ]

    # # primitives_lih = primitives_lithium + primitives_hydrogen

    # Li_1s = [
    #     (0.7946504870, 0.4446345422),
    #     (2.936200663, 0.5353281423),
    #     (16.11957475, 0.1543289673),
    #     ]

    # Li_2s = [
    #     (0.04808867840, 0.7001154689),
    #     (0.1478600533, 0.3995128261),
    #     (0.6362897469, -0.09996722919),
    #     ]

    # H_1s = [
    #     (0.109818, 0.444635),
    #     (0.405771, 0.535328),
    #     (2.227660, 0.154329),
    #     ]


    # def norm_prim(primitives, zeta=1.0):

    #     basis = []
    #     for alpha, d in primitives:
    #         alpha_scaled = alpha * (zeta ** 2)
    #         norm_factor = (2.0 * alpha_scaled / np.pi) ** 0.75
    #         d_scaled = d * norm_factor
    #         basis.append((alpha_scaled, d_scaled))
    
    #     return basis

    # Li_1s_norm = norm_prim(Li_1s)
    # Li_2s_norm = norm_prim(Li_2s)
    # H_1s_norm = norm_prim(H_1s, 1.24)

    # primitives_lih = [Li_1s_norm, Li_2s_norm, H_1s_norm]
    # # primitives_lih = [Li_1s, Li_2s, H_1s]

    # pos_lih = np.array([[0,0,0],[0,0,0],[1.6,0,0]])
    # # pos = [
    # #     np.array([0.0]),  # Li
    # #     np.array([0.0]),  # Li again (for 2s)
    # #     np.array([1.6]),  # H
    # # ]
    
    # Z_nuc = [3,1]
    # R_nuc = np.array([[0.0, 0.0, 0.0], [1.6, 0.0, 0.0]])
    # Z_lih = (3.0, 3.0, 1.0)  # match order of basis centers
    # n_elec_lih = 4

    # print("Primitives lih: ", primitives_lih)

    # scf_rhf(primitives_lih, pos=pos_lih, R=1.6, Z=Z_lih, n_elec=n_elec_lih, R_nuc=R_nuc, Z_nuc=Z_nuc)










