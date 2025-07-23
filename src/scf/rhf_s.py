import numpy as np
from scipy.linalg import eigh, fractional_matrix_power

from src.integrals.s_integrals import build_integral_arrays, canonical_eri_key


def scf_rhf(
        primitives: list[list[tuple[float, float]]],
        pos: np.ndarray, R: float, Z: tuple, n_elec: int,
        R_nuc: np.ndarray, Z_nuc: list, molecule,
        max_iter=50, conv_tol=1e-6, verbose=1
        ) -> dict:
    """
    Restricted Hartree-Fock SCF calculation for diatomic molecules.
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
    S, T, V, H_core, eri_dict = build_integral_arrays(primitives, pos, Z)

   
    # Number of basis functions and electrons
    nbf = S.shape[0]
    n_occ = n_elec // 2  # Number of occupied orbitals (doubly occupied)
    
    if verbose >= 1:
        print(f"Number of basis functions: {nbf}")
        print(f"Number of electrons: {n_elec}")
        print(f"Number of occupied orbitals: {n_occ}")
        print(f"Bond distance: {R:.3f} au")
        # print("-" * 50)
    
    # Symmetric orthogonalization: X = S^(-1/2)
    # This transforms the basis to an orthonormal one
    try:
        X = fractional_matrix_power(S, -0.5)
    except np.linalg.LinAlgError:
        # Fallback: canonical orthogonalization
        eigvals, eigvecs = eigh(S)
        X = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T


    # Initial guess: core Hamiltonian
    P = np.zeros_like(S)
    E_old = 0.0
    
    # SCF iterations
    for iteration in range(max_iter):
        
        # Build Fock matrix
        F = build_fock_matrix_sparse(H_core, P, eri_dict)
        
        # Calculate electronic energy
        E_elec = np.sum(P * H_core) + 0.5 * np.sum(P * (F - H_core)) # More numerically stable       

        # Nuclear repulsion energy (generalized for polyatomics)
        E_nuc = sum(Z_nuc[i]*Z_nuc[j] / np.linalg.norm(R_nuc[i]-R_nuc[j])
                    for i in range(len(Z_nuc))
                    for j in range(i+1, len(Z_nuc)))

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
        'kinetic_matrix': T,
        'nuclear_matrix': V,
        'core_hamiltonian': H_core,
        'iterations': iteration + 1,
        'converged': rms_change < conv_tol,
        'eri_tensor': eri_dict,
        'molecule': molecule
    }
    
    if verbose >= 1:
        print_final_results(results)
    
    return

def compute_nuclear_repulsion_energy(Z_nuc, R_nuc):
    E_nuc = 0.0
    for i in range(len(Z_nuc)):
        for j in range(i + 1, len(Z_nuc)):
            Rij = np.linalg.norm(np.array(R_nuc[i]) - np.array(R_nuc[j]))
            if Rij > 1e-12:
                E_nuc += Z_nuc[i] * Z_nuc[j] / Rij
    return E_nuc

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

def print_final_results(results):
    """Print formatted final SCF results"""
    print("\n" + "="*60)
    print(f"FINAL SCF {results['molecule']} RESULTS")
    print("="*60)

    print(f"{'Electronic energy:':<25}{results['energy_electronic']:>12.6f}  Ha")
    print(f"{'Nuclear repulsion:':<25}{results['energy_nuclear']:>12.6f}  Ha")
    print(f"{'Total energy:':<25}{results['energy_total']:>12.6f}  Ha")
    print(f"{'SCF iterations:':<25}{results['iterations']:>12d}")
    print(f"{'Converged:':<25}{str(results['converged']):>12}")
    
    # print("\nOrbital energies (Ha):")
    # for i, energy in enumerate(results['orbital_energies']):
    #     print(f"  ε_{i+1} = {energy:>10.6f}")
    
    # print("\nOrbital Coefficients (C):")
    # C = results['orbital_coefficients']
    # for i in range(C.shape[1]):
    #     coeffs = "  ".join(f"{val:>8.4f}" for val in C[:, i])
    #     print(f"  Orbital {i+1:<2}: {coeffs}")

    print("S matrix:\n", results['overlap_matrix'])
    print("T matrix:\n", results['kinetic_matrix'])
    print("V matrix:\n", results['nuclear_matrix'])
    print("P matrix:\n", results['density_matrix'])
    print("C matrix:\n", results['orbital_coefficients'])
    print("H matrix:\n", results['core_hamiltonian'])

    eri = results['eri_tensor']
    eri_size = len(eri)

    # print("ERI tensor shape:\n", eri_size)
    # if 'eri_tensor' in results:
    #     print("\nUnique Electron Repulsion Integrals (ERIs):")
    #     eri_dict = results['eri_tensor']
    #     for key, value in eri_dict.items():
    #         munu, lamsig = key  # Unpack the canonical key
    #         print(f"({munu}|{lamsig}): {value:.6f}")

   
