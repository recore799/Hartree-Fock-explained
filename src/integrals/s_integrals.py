import numpy as np
from scipy.special import erf
from collections import defaultdict


def boys_function(t: float) -> float:
    """
    Zero-order Boys function F0(t).
    Uses Taylor expansion for small t.
    """
    if t < 1e-6:
        return 1.0 - t/3.0
    return 0.5 * np.sqrt(np.pi/t) * erf(np.sqrt(t))



def kinetic_primitive(alpha: float, beta: float, R_ab: float) -> float:
    """
    Kinetic energy integral between two s-type primitives:
    T = q * (3 - 2*q*R_ab^2) * (pi/p)^(3/2) * exp(-q*R_ab^2)
    where p=alpha+beta, q=alpha*beta/p.
    """
    p = alpha + beta
    q = alpha * beta / p
    return q * (3.0 - 2.0*q*R_ab**2) * (np.pi / p)**1.5 * np.exp(-q * R_ab**2)


def nuclear_attraction_primitive(alpha: float, beta: float, R_ab: float,
                                 R_pc: float, Z_c: float) -> float:
    """
    Nuclear attraction integral ⟨g_alpha| -Z_c/|r-R_c| |g_beta⟩
    R_ab: distance between primitives centers A and B
    R_pc: distance from Gaussian product center P to nucleus C
    """
    p = alpha + beta
    q = alpha * beta / p
    prefac = -Z_c * 2 * np.pi / p * np.exp(-q * R_ab**2)
    return prefac * boys_function(p * R_pc**2)


def eri_primitive(alpha: float, beta: float,
                  gamma: float, delta: float,
                  R_ab: float, R_cd: float, R_pq: float) -> float:
    """
    Two-electron repulsion integral (alpha beta|gamma delta):
    = 2*pi^2.5/(p*q*sqrt(p+q)) * exp(-q_ab*R_ab^2 - q_cd*R_cd^2)
      * F0(p*q/(p+q)*R_pq^2)
    where p=alpha+beta, q=gamma+delta, q_ab=alpha*beta/p, q_cd=gamma*delta/q.
    """
    p = alpha + beta
    q = gamma + delta
    q_ab = alpha * beta / p
    q_cd = gamma * delta / q
    prefac = 2.0 * np.pi**2.5 / (p * q * np.sqrt(p + q))
    exp_term = np.exp(-q_ab * R_ab**2 - q_cd * R_cd**2)
    t = p * q / (p + q) * R_pq**2
    return prefac * exp_term * boys_function(t)


def overlap_primitive(alpha: float, beta: float, R_ab2: float) -> float:
    """
    Overlap integral between two s-type Gaussian primitives:
    S = (pi/(alpha+beta))^(3/2) * exp(-alpha*beta/(alpha+beta) * R_ab^2)
    """
    p = alpha + beta
    return (np.pi / p)**1.5 * np.exp(-alpha * beta * R_ab2 / p)

def compute_overlap_matrix(
        primitives: list[list[tuple[float, float]]],
        pos: np.ndarray) -> np.ndarray:
    """
    Compute overlap matrix with exact diagonal and optimized symmetry.
    """
    nbf = len(primitives)
    S = np.zeros((nbf, nbf))
    
    for mu in range(nbf):
        S[mu, mu] = 1.0  # Enforce exact normalization
        for nu in range(mu+1,nbf):  # Upper triangle only
            R2 = np.sum((pos[mu] - pos[nu])**2)
            overlap = 0.0
            for alpha_i, d_i in primitives[mu]:
                for alpha_j, d_j in primitives[nu]:
                    overlap += d_i * d_j * overlap_primitive(alpha_i, alpha_j, R2)
            S[mu, nu] = overlap
    
    # Mirror upper to lower triangle
    S = S + S.T - np.diag(S.diagonal())
    return S

def compute_kinetic_matrix(
        primitives: list[list[tuple[float, float]]],
        pos: np.ndarray) -> np.ndarray:
    nbf = len(primitives)
    T = np.zeros((nbf, nbf))
    for mu in range(nbf):
        for nu in range(nbf):
            R_mu_nu = np.linalg.norm(pos[mu] - pos[nu])
            for alpha_i, d_i in primitives[mu]:
                for alpha_j, d_j in primitives[nu]:
                    T[mu, nu] += d_i * d_j * kinetic_primitive(alpha_i, alpha_j, R_mu_nu)
    return T

def compute_nuclear_attraction_matrix(
    primitives: list[list[tuple[float, float]]],
    pos: np.ndarray, 
    Z: tuple[float, ...]
) -> np.ndarray:
    """
    Compute the nuclear attraction matrix for a molecule with arbitrary nuclei.
    Notes
    -----
    - Uses Gaussian product theorem to combine primitives.
    - Order of `pos` and `Z` must match 
    """
    nbf = len(primitives)
    V = np.zeros((nbf, nbf))

    for mu in range(nbf):
        for nu in range(nbf):
            # Distance between basis function centers (3D)
            R_mu_nu = np.linalg.norm(pos[mu] - pos[nu])

            for alpha_i, d_i in primitives[mu]:
                for alpha_j, d_j in primitives[nu]:
                    p = alpha_i + alpha_j
                    # Product Gaussian center (3D)
                    P_pos = (alpha_i*pos[mu] + alpha_j*pos[nu]) / p

                    # Sum contributions from all nuclei
                    for R_i, Z_i in zip(pos, Z):
                        R_pc = np.linalg.norm(P_pos - R_i) # 3D distance
                        V[mu,nu] += d_i * d_j * nuclear_attraction_primitive(
                            alpha_i, alpha_j, R_mu_nu, R_pc, Z_i)
    return V

def canonical_eri_key(mu, nu, lam, sig):
    """Create canonical representation preserving pair symmetry"""
    # Sort within pairs
    pair1 = tuple(sorted((mu, nu)))
    pair2 = tuple(sorted((lam, sig)))
    # Sort between pairs
    if pair1 <= pair2:
        return pair1 + pair2
    else:
        return pair2 + pair1

def compute_eri_tensor_sparse(primitives, pos):
    eri_dict = defaultdict(float)
    nbf = len(primitives)
    computed_keys = set()

    for mu in range(nbf):
        for nu in range(nbf):
            for lam in range(nbf):
                for sig in range(nbf):
                    # Get canonical representation
                    key = canonical_eri_key(mu, nu, lam, sig)
                    
                    # Skip if we've already computed this symmetry class
                    if key in computed_keys:
                        continue
                    
                    # Mark this symmetry class as computed
                    computed_keys.add(key)
                    
                    # Compute the integral value
                    eri_val = 0.0
                    for ai, di in primitives[mu]:
                        for aj, dj in primitives[nu]:
                            for ak, dk in primitives[lam]:
                                for al, dl in primitives[sig]:
                                    p = ai + aj
                                    q = ak + al
                                    P = (ai * pos[mu] + aj * pos[nu]) / p
                                    Q = (ak * pos[lam] + al * pos[sig]) / q
                                    R_mn = np.linalg.norm(pos[mu] - pos[nu])
                                    R_ls = np.linalg.norm(pos[lam] - pos[sig])
                                    R_pq = np.linalg.norm(P - Q)

                                    eri_val += (
                                        di * dj * dk * dl * 
                                        eri_primitive(ai, aj, ak, al, R_mn, R_ls, R_pq)
                                    )
                    
                    # Store the full value under the canonical key
                    eri_dict[key] = eri_val
                    
    return eri_dict

def build_integral_arrays(primitives, pos, Z):
    S = compute_overlap_matrix(primitives, pos)
    T = compute_kinetic_matrix(primitives, pos)
    V = compute_nuclear_attraction_matrix(primitives, pos, Z)
    H_core = T + V
    eri_dict = compute_eri_tensor_sparse(primitives, pos)
    return S, T, V, H_core, eri_dict



if __name__ == "__main__":

    def compare_eris_dense_sparse(eri_dense: np.ndarray, eri_sparse: dict, atol: float = 1e-8):
        """
        Compare each (μν|λσ) element in the dense ERI tensor with the value from the sparse dictionary.

    """
        nbf = eri_dense.shape[0]
        max_diff = 0.0
        n_diffs = 0

        for mu in range(nbf):
            for nu in range(nbf):
                for lam in range(nbf):
                    for sig in range(nbf):
                        dense_val = eri_dense[mu, nu, lam, sig]
                        key = canonical_eri_key(mu, nu, lam, sig)
                        sparse_val = eri_sparse[key]

                        diff = abs(dense_val - sparse_val)
                        if diff > atol:
                            print(f"Mismatch at ({mu},{nu},{lam},{sig}): dense = {dense_val:.10e}, "
                                  f"sparse = {sparse_val:.10e}, diff = {diff:.2e}")
                            n_diffs += 1
                            max_diff = max(max_diff, diff)

        if n_diffs == 0:
            print("✅ Dense and sparse ERI representations agree within tolerance.")
        else:
            print(f"❗ Found {n_diffs} mismatches. Max difference = {max_diff:.2e}")

    from sto3g_basis import build_sto3g_basis

    pos = [0.0, 1.4632]

    sto3g_h = build_sto3g_basis(zeta=1.24)    # Hydrogen basis
    sto3g_he = build_sto3g_basis(zeta=2.095)  # Helium basis
    primitives = [sto3g_he, sto3g_h]  # [He, H]


    eri_dense = compute_eri_tensor(primitives, pos)
    eri_sparse = compute_eri_tensor_sparse(primitives, pos)

    compare_eris_dense_sparse(eri_dense, eri_sparse)




