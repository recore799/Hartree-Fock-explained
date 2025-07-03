import numpy as np
from math import prod, sqrt, pi, exp
from collections import defaultdict
from scipy.special import erf
from collections import defaultdict

# ---- Symmetry utilities (ERIs) ----
def pack_index(mu, nu):
    # Pack two indices into one unique key (assuming mu >= nu)
    return mu * (mu + 1) // 2 + nu

def get_canonical_key(mu, nu, lam, sig):
    # Exploit 8-fold ERI symmetry: (ij|kl) == (ji|kl) == ...
    munu = pack_index(max(mu, nu), min(mu, nu))
    lamsig = pack_index(max(lam, sig), min(lam, sig))
    # Order the two pairs to enforce (munu >= lamsig)
    return (munu, lamsig) if munu >= lamsig else (lamsig, munu)

def double_factorial(n: int) -> int:
    """Compute double factorial n!! = n*(n-2)*(n-4)*...*1 (or 2)"""
    if n < 0:
        return 1
    return prod(range(n, 0, -2))


def boys_sequence(max_m: int, T: float) -> np.ndarray:
    """
    Compute Boys function F_0(T) to F_{max_m}(T) using stable downward recursion.
    
    Args:
        max_m: Maximum order needed (returns F[0]...F[max_m])
        T: Argument to Boys function
    
    Returns:
        Array of F[0], F[1], ..., F[max_m]
    """
    if T < 1e-15:  # Taylor series for small T
        return np.array([1/(2*m + 1) - T/(2*m + 3) for m in range(max_m + 1)])
    
    # Initialize array with F[0] to F[max_m]
    F = np.zeros(max_m + 1)
    
    # Base case for F[0]
    sqrt_T = sqrt(T)
    F[0] = sqrt(pi)/2 * erf(sqrt_T)/sqrt_T if T > 0 else 1.0
    
    if max_m == 0:
        return F
    
    # Estimate needed starting point (empirically optimized)
    n_start = max_m + int(10 + 2*T)  # Reduced safety margin
    
    # Temporary storage for downward recursion
    F_temp = np.zeros(n_start + 2)
    F_temp[n_start + 1] = 1/(2*(n_start + 1) + 1)  # Approximate F[n_start+1]
    
    # Downward recursion to fill F_temp
    for m in range(n_start, -1, -1):
        F_temp[m] = (2*T*F_temp[m + 1] + exp(-T))/(2*m + 1)
    
    # Copy required values (F[0] is already set)
    F[1:max_m + 1] = F_temp[1:max_m + 1]
    
    return F


def compute_primitive_parameters(a_prim, b_prim, c_prim, d_prim):
    """
    Given four primitives, return a dict with all deterministic parameters:
    """

    zeta_a, A, (la,ma,na) = a_prim
    zeta_b, B, (lb,mb,nb) = b_prim
    zeta_c, C, (lc,mc,nc) = c_prim
    zeta_d, D, (ld,md,nd) = d_prim

    # Gaussian product theorem
    zeta = zeta_a + zeta_b
    eta = zeta_c + zeta_d
    xi = (zeta_a*zeta_b) / (zeta_a + zeta_b)
    P = (zeta_a*A + zeta_b*B) / zeta
    Q = (zeta_c*C + zeta_d*D) / eta
    W = (zeta*P + eta*Q) / (zeta+eta)
    rho = zeta * eta / (zeta + eta)
    RPQ = P-Q
    T = rho * np.dot(RPQ, RPQ)

    # Boys function up to max needed order
    max_m = la + lb + lc + ld + ma + mb + mc + md + na + nb + nc + nd
    F = boys_sequence(max_m+10, T)

    # Compute K prefactor for (00|00)
    def K_func(zeta,zeta_p,R,R_p):
        RRp = R - R_p
        return 2**0.5 * (pi**(5/4)) / (zeta + zeta_p) * np.exp( - (zeta*zeta_p * np.dot(RRp, RRp)) / (zeta + zeta_p) )


    # Prefactors and vectors for recurrences
    ssss_coeff = K_func(zeta_a, zeta_b, A, B) * K_func(zeta_c, zeta_d, C, D) / sqrt(zeta+eta)

    RP_A = P-A
    RP_B = P-B
    RQ_C = Q-C
    RQ_D = Q-D
    RW_P = W-P
    RW_Q = W-Q


    return {
        'a_ang': (la, ma, na), 'b_ang': (lb, mb, nb),
        'c_ang': (lc, mc, nc), 'd_ang': (ld, md, nd),
        'zeta': zeta, 'eta': eta, 'rho': rho, 'xi': xi, 'A': A, 'B': B, 'C': C, 'D': D,
        'ssss_coeff': ssss_coeff,
        'RP_A': RP_A, 'RP_B': RP_B, 'RQ_C': RQ_C, 'RQ_D': RQ_D, 'RW_P': RW_P, 'RW_Q': RW_Q,
        'F': F, 'T': T
        }
# I tensor utilities
# set_ sets val to I[m][i] tensor element
# ret_ is for accessing said element

# (ss|ss)^(m)
def set_ssss(I,m, val):
    I[m][0,0,0, 0,0,0, 0,0,0, 0,0,0] = val
def ret_ssss(I,m):
    return I[m][0,0,0, 0,0,0, 0,0,0, 0,0,0]

# (ps|ss)^(m)
def set_psss(I,m,i, val):
    I[m][1 if i==0 else 0,
         1 if i==1 else 0,
         1 if i==2 else 0,
         0,0,0, 0,0,0, 0,0,0] = val
def ret_psss(I,m,i):
    return I[m][1 if i==0 else 0,
                1 if i==1 else 0,
                1 if i==2 else 0,
                0,0,0, 0,0,0, 0,0,0]
# (sp|ss)^(m)
def set_spss(I,m,j, val):
    I[m][1 if j==0 else 0,
         1 if j==1 else 0,
         1 if j==2 else 0,
         0,0,0, 0,0,0, 0,0,0] = val
def ret_spss(I,m,j):
    return I[m][1 if j==0 else 0,
                1 if j==1 else 0,
                1 if j==2 else 0,
                0,0,0, 0,0,0, 0,0,0]

# (ss|ps)^(m)
def set_ssps(I,m,k, val):
    I[m][1 if k==0 else 0,
         1 if k==1 else 0,
         1 if k==2 else 0,
         0,0,0, 0,0,0, 0,0,0] = val
def ret_ssps(I,m,k):
    return I[m][1 if k==0 else 0,
                1 if k==1 else 0,
                1 if k==2 else 0,
                0,0,0, 0,0,0, 0,0,0]

# (ss|sp)^(m)
def set_sssp(I,m,l, val):
    I[m][1 if l==0 else 0,
         1 if l==1 else 0,
         1 if l==2 else 0,
         0,0,0, 0,0,0, 0,0,0] = val
def ret_sssp(I,m,l):
    return I[m][1 if l==0 else 0,
                1 if l==1 else 0,
                1 if l==2 else 0,
                0,0,0, 0,0,0, 0,0,0]

# (ps|ps)^(m)
def set_psps(I,m,i,k, val):
    I[m][1 if i==0 else 0,
         1 if i==1 else 0,
         1 if i==2 else 0,
         0,0,0,
         1 if k==0 else 0,
         1 if k==1 else 0,
         1 if k==2 else 0,
         0,0,0] = val
def ret_psps(I,m,i,k):
    return I[m][1 if i==0 else 0,
                1 if i==1 else 0,
                1 if i==2 else 0,
                0,0,0,
                1 if k==0 else 0,
                1 if k==1 else 0,
                1 if k==2 else 0,
                0,0,0]

# (sp|ps)^(m)
def set_spps(I,m,j,k, val):
    I[m][0,0,0,
         1 if j==0 else 0,
         1 if j==1 else 0,
         1 if j==2 else 0,
         1 if k==0 else 0,
         1 if k==1 else 0,
         1 if k==2 else 0,
         0,0,0] = val
def ret_spps(I,m,j,k):
    return I[m][0,0,0,
                1 if j==0 else 0,
                1 if j==1 else 0,
                1 if j==2 else 0,
                1 if k==0 else 0,
                1 if k==1 else 0,
                1 if k==2 else 0,
                0,0,0]

# (ps|sp)^(m)
def set_pssp(I,m,i,l, val):
    I[m][1 if i==0 else 0,
         1 if i==1 else 0,
         1 if i==2 else 0,
         0,0,0,
         0,0,0,
         1 if l==0 else 0,
         1 if l==1 else 0,
         1 if l==2 else 0] = val
def ret_pssp(I,m,i,l):
    return I[m][1 if i==0 else 0,
                1 if i==1 else 0,
                1 if i==2 else 0,
                0,0,0,
                0,0,0,
                1 if l==0 else 0,
                1 if l==1 else 0,
                1 if l==2 else 0]

# (ss|pp)^(m)
def set_psps(I,m,k,l, val):
    I[m][0,0,0, 0,0,0,
         1 if k==0 else 0,
         1 if k==1 else 0,
         1 if k==2 else 0,
         1 if l==0 else 0,
         1 if l==1 else 0,
         1 if l==2 else 0] = val
def ret_sspp(I,m,k,l):
    return I[m][0,0,0, 0,0,0,
         1 if k==0 else 0,
         1 if k==1 else 0,
         1 if k==2 else 0,
         1 if l==0 else 0,
         1 if l==1 else 0,
         1 if l==2 else 0]

# (pp|ss)^(m)
def set_ppss(I,m,i,j, val):
    I[m][1 if i==0 else 0,
         1 if i==1 else 0,
         1 if i==2 else 0,
         1 if j==0 else 0,
         1 if j==1 else 0,
         1 if j==2 else 0,
         0,0,0, 0,0,0] = val
def ret_ppss(I,m,i,j):
    return I[m][1 if i==0 else 0,
                1 if i==1 else 0,
                1 if i==2 else 0,
                1 if j==0 else 0,
                1 if j==1 else 0,
                1 if j==2 else 0,
                0,0,0, 0,0,0]

# (pp|ps)^(m)
def set_ppps(I,m,i,j,k, val):
    I[m][1 if i==0 else 0,
         1 if i==1 else 0,
         1 if i==2 else 0,
         1 if j==0 else 0,
         1 if j==1 else 0,
         1 if j==2 else 0,
         1 if k==0 else 0,
         1 if k==1 else 0,
         1 if k==2 else 0,
         0,0,0] = val
def ret_ppps(I,m,i,j,k):
    return I[m][1 if i==0 else 0,
                1 if i==1 else 0,
                1 if i==2 else 0,
                1 if j==0 else 0,
                1 if j==1 else 0,
                1 if j==2 else 0,
                1 if k==0 else 0,
                1 if k==1 else 0,
                1 if k==2 else 0,
                0,0,0]

# (pp|pp)^(m)
def set_pppp(I,m,i,j,k,l, val):
    I[m][1 if i==0 else 0,
         1 if i==1 else 0,
         1 if i==2 else 0,
         1 if j==0 else 0,
         1 if j==1 else 0,
         1 if j==2 else 0,
         1 if k==0 else 0,
         1 if k==1 else 0,
         1 if k==2 else 0,
         1 if l==0 else 0,
         1 if l==1 else 0,
         1 if l==2 else 0] = val
def ret_pppp(I,m,i,j,k,l):
    return I[m][1 if i==0 else 0,
                1 if i==1 else 0,
                1 if i==2 else 0,
                1 if j==0 else 0,
                1 if j==1 else 0,
                1 if j==2 else 0,
                1 if k==0 else 0,
                1 if k==1 else 0,
                1 if k==2 else 0,
                1 if l==0 else 0,
                1 if l==1 else 0,
                1 if l==2 else 0]

# Recursion builders
def build_psss(I, RP_A, RW_P, max_m):
    for m in range(max_m):
        for i in range(3):
            val = (
                RP_A[i] * ret_ssss(I,m) +
                RW_P[i] * ret_ssss(I,m+1)
            )
            set_psss(I,m,i, val)

def build_spss(I, RP_B, RW_P, max_m):
    for m in range(max_m):
        for j in range(3):
            val = (
                RP_B[j] * ret_ssss(I,m) +
                RW_P[j] * ret_ssss(I,m+1)
            )
            set_spss(I,m,j, val)

def build_ssps(I, RQ_C, RW_Q, max_m):
    for m in range(max_m):
        for k in range(3):
            val = (
                RQ_C[k] * ret_ssss(I,m) +
                RW_Q[k] * ret_ssss(I,m+1)
            )
            set_ssps(I,m,k, val)

def build_sssp(I, RQ_D, RW_Q, max_m):
    for m in range(max_m):
        for l in range(3):
            val = (
                RQ_D[l] * ret_ssss(I,m) +
                RW_Q[l] * ret_ssss(I,m+1)
            )
            set_sssp(I,m,l, val)


            
def build_psps(I, RQ_C, RW_Q, zeta, eta, max_m):
    for m in range(max_m-1):
        for i in range(3):
            for k in range(3):
                val = (
                    RQ_C[k] * ret_psss(I,m,i) +
                    RW_Q[k] * ret_psss(I,m+1,i) +
                    (delta(i,k) / (2*(zeta+eta))) * ret_ssss(I,m+1)
                )
                set_psps(I,m,i,k, val) 

def build_spps(I, RQ_C, RW_Q, zeta, eta, max_m):
    for m in range(max_m - 1):
        for j in range(3):  # B center (second)
            for k in range(3):  # C center (third)
                val = (
                    RQ_C[k] * ret_spss(I,m,j) +
                    RW_Q[k] * ret_spss(I,m+1,j) +
                    delta(j,k)/(2*(zeta + eta)) * ret_ssss(I,m+1)
                )
                set_spps(I,m,j,k, val)

def build_pssp(I, RQ_D, RW_Q, zeta, eta, max_m):
    for m in range(max_m - 1):
        for i in range(3):  # A center (first)
            for l in range(3):  # D center (fourth)
                val = (
                    RQ_D[l] * ret_psss(I,m,i) +
                    RW_Q[l] * ret_psss(I,m+1,i) +
                    delta(i,l)/(2*(zeta + eta)) * ret_ssss(I,m+1)
                )
                set_pssp(I,m,i,l, val)

def build_sspp(I, RQ_D, RW_Q, rho, eta, max_m):
    for m in range(max_m - 1):
        for k in range(3):  # C center (third)
            for l in range(3):  # D center (fourth)
                val = (
                    RQ_D[l] * ret_ssps(I,m,k) +
                    RW_Q[l] * ret_ssps(I,m+1,k) +
                    delta(k,l)/(2*eta) * (
                        ret_ssss(I,m) - (rho/eta) * ret_ssss(I,m+1)
                    )
                )
                set_sspp(I,m,k,l, val)

def build_ppss(I, RP_B, RW_P, rho, zeta, max_m):
    for m in range(max_m-1):
        for i in range(3):
            for j in range(3):
                val = (
                    RP_B[j] * ret_psss(I,m,i) +
                    RW_P[j] * ret_psss(I,m+1,i) +
                    (delta(i,j)/(2*zeta)) * ( ret_ssss(I,m) - (rho/zeta) * ret_ssss(I,m+1))
                )
                set_ppss(I,m,i,j, val)

def build_ppps(I, RQ_C, RW_Q, zeta, eta, max_m):
    for m in range(max_m-2):
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    # (p_i p_j, p_k s)^(m) =
                    # (Q_k - C_k)(p_i p_j, ss)^(m) +
                    # (W_k - Q_k)(p_i p_j, ss)^(m+1)
                    # [1/(2(zeta+eta))]*(delta(ik)(s p_j,ss)^(m+1) + delta(jk)(p_i s,ss)^(m+1) )
                    val = (
                        RQ_C[k] * ret_ppss(I,m,i,j) +
                        RW_Q[k] * ret_ppss(I,m+1,i,j) +
                        (delta(i,k) * ret_spss(I,m+1,j) +
                         delta(j,k) * ret_psss(I,m+1,i) / (2*(zeta+eta)))
                    )
                    set_ppps(I,m,i,j,k, val)

def build_pppp(I, RQ_D, RW_Q, zeta, eta, rho, max_m):
    for m in range(max_m-3):
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        val = (
                            RQ_D[l] * ret_ppps(I, m, i, j, k) + RW_Q[l] * ret_ppps(I, m+1, i, j, k) +
                            (delta(i,l) * ret_spps(I, m+1, j, k) + delta(j,l) * ret_psps(I, m+1, i, k)) / (2 * (zeta + eta)) +
                            (delta(k,l) / (2*eta)) * (ret_ppss(I, m, i, j) - (rho/eta) * ret_ppss(I, m+1, i, j))
                        )
                        set_pppp(I, m, i, j, k, l, val)


def needs_recursion(params):
    la, ma, na = params['a_ang']
    lb, mb, nb = params['b_ang']
    lc, mc, nc = params['c_ang']
    ld, md, nd = params['d_ang']

    def total(lmn): return sum(lmn)

    need = {
        "psss": total((la, ma, na)) > 0,
        "spss": total((lb, mb, nb)) > 0,
        "ssps": total((lc, mc, nc)) > 0,
        "sssp": total((ld, md, nd)) > 0,
    }

    # Precompute derived needs
    need["ppss"] = need["psss"] and need["spss"]
    need["ppps"] = need["ppss"] and need["ssps"]
    need["pppp"] = need["ppps"] and need["sssp"]

    return need

# Kronecker delta
def delta(i, j): return 1.0 if i == j else 0.0


def dp_primitive_eri(params):
    # Unpack parameters
    la, ma, na = params['a_ang']
    lb, mb, nb = params['b_ang']
    lc, mc, nc = params['c_ang']
    ld, md, nd = params['d_ang']
    zeta, eta, rho = params['zeta'], params['eta'], params['rho']
    RP_A, RP_B = params['RP_A'], params['RP_B']
    RQ_C, RQ_D = params['RQ_C'], params['RQ_D']
    RW_P, RW_Q = params['RW_P'], params['RW_Q']
    F = params['F']
    ssss_coeff = params['ssss_coeff']

    # Determine whether each vertical recursion layer is needed
    need = needs_recursion(params)

    # max order for (ss,ss)^(m)
    max_m = la + ma + na + lb + mb + nb + lc + mc + nc + ld + md + nd

    # Allocate full tensor
    I = [np.zeros((2,)*12) for _ in range(max_m + 1)]

    # Base case up to max order [(ss,ss)^(m)]
    for m in range(max_m + 1):
        I[m][0,0,0, 0,0,0, 0,0,0, 0,0,0] = ssss_coeff * F[m]
        # print(f"F[{m}] = {F[m]}")

    if need["psss"]:  build_psss(I, RP_A, RW_P, max_m)
    if need["spss"]:  build_spss(I, RP_B, RW_P, max_m)
    if need["ssps"]:  build_ssps(I, RQ_C, RW_Q, max_m)
    if need["sssp"]:  build_sssp(I, RQ_D, RW_Q, max_m)
    if need["ppss"]:  build_ppss(I, RP_B, RW_P, rho, zeta, max_m)
    if need["ppps"]:  build_ppps(I, RQ_C, RW_Q, zeta, eta, max_m)
    if need["pppp"]:
        build_spps(I, RQ_C, RW_Q, zeta, eta, max_m)
        build_psps(I, RQ_C, RW_Q, zeta, eta, max_m)
        build_pppp(I, RQ_D, RW_Q, zeta, eta, rho, max_m)

    return I

def gaussian_norm(zeta: float, angmom: tuple[int,int,int]) -> float:
    """Compute normalization constant for a primitive Gaussian."""
    l, m, n = angmom
    prefactor = (2*zeta/np.pi)**0.75 * (4*zeta)**((l+m+n)/2)
    dfactor = np.sqrt(double_factorial(2*l-1) * double_factorial(2*m-1) * double_factorial(2*n-1))
    return prefactor * dfactor



def compute_eri_element(mu, nu, lam, sig, basis_set):
    """
    Compute the electron repulsion integral (μν|λσ) over basis functions μ, ν, λ, σ.

    Args:
        mu, nu, lam, sig (int): Indices of the basis functions.
        basis_set (dict[int -> BasisFunction]): Dictionary of basis functions.

    Returns:
        float: Value of the ERI.
    """
    ci = basis_set[mu]
    cj = basis_set[nu]
    ck = basis_set[lam]
    cl = basis_set[sig]

    eri = 0.0

    for a_prim in ci.primitives:
        for b_prim in cj.primitives:
            for c_prim in ck.primitives:
                for d_prim in cl.primitives:
                    # Unpack primitives
                    zeta_a, coeff_a, center_a, ang_a = a_prim.zeta, a_prim.coeff, a_prim.center, a_prim.angmom
                    zeta_b, coeff_b, center_b, ang_b = b_prim.zeta, b_prim.coeff, b_prim.center, b_prim.angmom
                    zeta_c, coeff_c, center_c, ang_c = c_prim.zeta, c_prim.coeff, c_prim.center, c_prim.angmom
                    zeta_d, coeff_d, center_d, ang_d = d_prim.zeta, d_prim.coeff, d_prim.center, d_prim.angmom

                    # Compute normalization
                    norm = (
                        gaussian_norm(zeta_a, ang_a) *
                        gaussian_norm(zeta_b, ang_b) *
                        gaussian_norm(zeta_c, ang_c) *
                        gaussian_norm(zeta_d, ang_d)
                    )

                    # Compute deterministic parameters for OS recursion
                    params = compute_primitive_parameters(
                        (zeta_a, center_a, ang_a),
                        (zeta_b, center_b, ang_b),
                        (zeta_c, center_c, ang_c),
                        (zeta_d, center_d, ang_d)
                    )

                    # Compute primitive ERI using OS recursion
                    val_tensor = dp_primitive_eri(params)

                    # Access the specific angular momentum component
                    la, ma, na = ang_a
                    lb, mb, nb = ang_b
                    lc, mc, nc = ang_c
                    ld, md, nd = ang_d

                    val = val_tensor[0][la, ma, na, lb, mb, nb, lc, mc, nc, ld, md, nd]

                    # Accumulate total ERI with weights and normalization
                    total_weight = coeff_a * coeff_b * coeff_c * coeff_d
                    eri += total_weight * val * norm

    return eri


def compute_eri_tensor_sparse(basis_set):
    """
    Compute sparse ERI tensor using Obara–Saika and packed index canonical keys.
    Args:
        basis_set (dict[int -> BasisFunction])
    Returns:
        eri_dict (defaultdict[tuple[int, int], float])
    """
    nbf = len(basis_set)
    eri_dict = defaultdict(float)
    computed_keys = set()

    for mu in range(nbf):
        for nu in range(nbf):
            for lam in range(nbf):
                for sig in range(nbf):
                    key = get_canonical_key(mu, nu, lam, sig)
                    if key in computed_keys:
                        continue
                    computed_keys.add(key)

                    val = compute_eri_element(mu, nu, lam, sig, basis_set)
                    eri_dict[key] = val

    return eri_dict

####################################################################################################
####################################################################################################
####################################################################################################

def set_ss(I, val):
    I[0,0,0, 0,0,0] = val
def ret_ss(I):
    return I[0,0,0, 0,0,0]


def set_ps(I, i, val):
    I[1 if i==0 else 0,
      1 if i==1 else 0,
      1 if i==2 else 0,
      0,0,0] = val
def ret_ps(I, i):
    return I[1 if i==0 else 0,
             1 if i==1 else 0,
             1 if i==2 else 0,
             0,0,0]

def set_sp(I, j, val):
    I[0,0,0,
      1 if j==0 else 0,
      1 if j==1 else 0,
      1 if j==2 else 0] = val
def ret_sp(I, j):
    return I[0,0,0,
             1 if j==0 else 0,
             1 if j==1 else 0,
             1 if j==2 else 0]

def set_pp(I, i, j, val):
    I[1 if i==0 else 0,
      1 if i==1 else 0,
      1 if i==2 else 0,
      1 if j==0 else 0,
      1 if j==1 else 0,
      1 if j==2 else 0] = val
    
def ret_pp(I, i, j):
    return I[1 if i==0 else 0,
             1 if i==1 else 0,
             1 if i==2 else 0,
             1 if j==0 else 0,
             1 if j==1 else 0,
             1 if j==2 else 0]
    
def build_ps(I, RP_A):
    for i in range(3):
        val = RP_A[i] * ret_ss(I)
        set_ps(I, i, val)

def build_sp(I, RP_B):
    for j in range(3):
        val = RP_B[j] * ret_ss(I)
        set_sp(I, j, val)

def build_pp(I, zeta, RP_B):
    for i in range(3):
        for j in range(3):
            val = RP_B[j] * ret_ps(I, i) + delta(i,j) / (2*zeta) * ret_ss(I)
            set_pp(I, i, j, val)

    
    
def dp_primitive_overlap(params):
    """
    Compute 1-electron overlap integrals using Obara–Saika recursion.
    Returns: I[3][3] where indices: [0] = s, [1] = px, [2] = py, [3] = pz
    """
    A = params['A']
    B = params['B']
    RP_A = params['RP_A']
    RP_B = params['RP_B']
    zeta = params['zeta']
    xi = params['xi']

    I = np.zeros((2,)*6)

    # Base case (s|s)
    RAB2 = np.dot(A-B, A-B)
    val = (np.pi / zeta) ** (1.5) * np.exp(-xi * RAB2)
    set_ss(I, val)

    # (p|s) and (s|p)
    build_ps(I, RP_A)
    build_sp(I, RP_B)
    # (p|p)
    build_pp(I, zeta, RP_B)

    return I

def compute_overlap_element(mu, nu, basis_set):
    """
    Compute the overlap integral S_{μν} using OS recursion.
    """
    ci = basis_set[mu]
    cj = basis_set[nu]
    S = 0.0

    for a_prim in ci.primitives:
        for b_prim in cj.primitives:
            zeta_a, coeff_a, center_a, ang_a = a_prim.zeta, a_prim.coeff, a_prim.center, a_prim.angmom
            zeta_b, coeff_b, center_b, ang_b = b_prim.zeta, b_prim.coeff, b_prim.center, b_prim.angmom

            norm = (
                gaussian_norm(zeta_a, ang_a) *
                gaussian_norm(zeta_b, ang_b)
            )

            params = compute_primitive_parameters(
                (zeta_a, center_a, ang_a),
                (zeta_b, center_b, ang_b),
                (zeta_a, center_a, ang_a),
                (zeta_b, center_b, ang_b)
            )

            val_smat = dp_primitive_overlap(params)

            la, ma, na = ang_a
            lb, mb, nb = ang_b

            # index shift: (s=0) → 0, (p=1) → 1, etc.
            val = val_smat[la, ma, na, lb, mb, nb]
            total_weight = coeff_a * coeff_b
            S += total_weight * val * norm

    return S

def compute_overlap_matrix(basis_set):
    nbf = len(basis_set)
    S = np.zeros((nbf, nbf))

    for mu in range(nbf):
        for nu in range(mu + 1):  # exploit symmetry
            val = compute_overlap_element(mu, nu, basis_set)
            S[mu, nu] = S[nu, mu] = val

    return S


# ——————————————————————————————————————————————
# 1) Primitive‐level kernels
# ——————————————————————————————————————————————

def build_psT(I, I2, xi, RP_A):
    for i in range(3):
        ss = ret_ss(I2)
        val = RP_A[i] * ret_ss(I) + 2*xi*ss
        set_ps(I, i, val)

def build_spT(I, I2, xi, RP_B):
    for j in range(3):
        ss = ret_ss(I2)
        val = RP_B[j] * ret_ss(I) + 2*xi*ss
        set_ps(I, j, val)

def build_ppT(I, I2, zeta, xi, RP_B):
    for i in range(3):
        for j in range(3):
            pp = ret_pp(I2, i, j)
            val = RP_B[j] * ret_pp(I, i, j) + delta(i, j) / (2*zeta) * ret_ss(I) + 2*xi*pp
            set_pp(I, i, j, val)

        

def dp_primitive_kinetic(params):
    """
    Obara–Saika vertical recursion for kinetic integrals:
      ⟨μ| -½∇² |ν⟩
    Assumes set_ssT, set_psT, set_spT, set_ppT are defined,
    as are ret_ss, ret_ps, etc., and delta(i,j).
    """
    # Unpack what you need
    A, B      = params['A'], params['B']
    RP_A, RP_B= params['RP_A'], params['RP_B']
    zeta       = params['zeta']
    xi         = params['rho']
    # 2×2×2×2… tensor of shape (2,2,2,2,2,2) for s/p on μ and s/p on ν
    I1 = np.zeros((2,)*6)
    I2 = np.zeros((2,)*6)

    # Base (s|T|s) kinetic:
    # FIX: uses explicit (s||s), resuse from S matrix instead
    RAB2 = np.dot(A - B, A - B)

    valS = (np.pi / zeta) ** (1.5) * np.exp(-xi * RAB2)
    set_ss(I2, valS)


    for i in range(3):
        val = RP_A[i] * ret_ss(I2)
        set_ps(I2, i, val)

    for j in range(3):
        val = RP_B[j] * ret_ss(I2)
        set_sp(I2, j, val)

    for i in range(3):
        for j in range(3):
            val = RP_B[j] * ret_ps(I2, i) + delta(i,j) / (2*zeta) * ret_ss(I2)
            set_pp(I2, i, j, val)


    valT = xi * (3 - 2*xi*RAB2) * ret_ss(I2)
    set_ss(I1, valT)

    build_psT(I1, I2, xi, RP_A)
    build_spT(I1, I2, xi, RP_B)
    build_ppT(I1, I2, zeta, xi, RP_B)
    
    return I1


def dp_primitive_nuclear(params, Cn, Zn):
    """
    Obara–Saika vertical recursion for nuclear‐attraction to nucleus at Cn with charge Zn:
      ⟨μ| -Z/|r-Cn| |ν⟩
    Assumes set_ssV, set_psV, set_spV, set_ppV are defined.
    """
    A, B      = params['A'], params['B']
    RP_A, RP_B= params['RP_A'], params['RP_B']
    zeta       = params['zeta']
    rho        = params['rho']
    RPQ        = params['P'] - Cn   # P from compute_primitive_parameters
    T          = rho * np.dot(RPQ, RPQ)

    # compute Boys F[0], F[1] from params['F']
    F = params['F']

    I = np.zeros((2,)*6)

    # Base (s|s) nuclear:
    # Vss = -Z * 2π/ζ * F₀(T)
    Vss = -Zn * 2*np.pi/zeta * F[0]
    set_ssV(I, Vss)

    # (p|s) and (s|p)
    for i in range(3):
        set_psV(I, i, RP_A[i]*Vss - Zn*(2*np.pi/zeta)*RPQ[i]*F[1])
        set_spV(I, i, RP_B[i]*Vss - Zn*(2*np.pi/zeta)*RPQ[i]*F[1])

    # (p|p)
    for i in range(3):
        for j in range(3):
            # Mixed-term and delta-term from OS formula
            term1 = RP_B[j]*ret_psV(I,i)
            term2 = -Zn*(2*np.pi/zeta)*(RPQ[j]*ret_psssV(I,i) + RPQ[i]*ret_spsV(I,j))
            term3 = delta(i,j)*(-Zn*(2*np.pi/zeta))*(F[0] - (rho/zeta)*F[1])
            set_ppV(I, i, j, term1 + term2 + term3)

    return I

# ——————————————————————————————————————————————
# 2) Contracted‐element builders
# ——————————————————————————————————————————————
def compute_kinetic_element(mu, nu, basis_set):
    """
    Compute the overlap integral S_{μν} using OS recursion.
    """
    ci = basis_set[mu]
    cj = basis_set[nu]
    T = 0.0

    for a_prim in ci.primitives:
        for b_prim in cj.primitives:
            zeta_a, coeff_a, center_a, ang_a = a_prim.zeta, a_prim.coeff, a_prim.center, a_prim.angmom
            zeta_b, coeff_b, center_b, ang_b = b_prim.zeta, b_prim.coeff, b_prim.center, b_prim.angmom

            norm = (
                gaussian_norm(zeta_a, ang_a) *
                gaussian_norm(zeta_b, ang_b)
            )

            params = compute_primitive_parameters(
                (zeta_a, center_a, ang_a),
                (zeta_b, center_b, ang_b),
                (zeta_a, center_a, ang_a),
                (zeta_b, center_b, ang_b)
            )

            val_tmat = dp_primitive_kinetic(params)

            la, ma, na = ang_a
            lb, mb, nb = ang_b

            val = val_tmat[la, ma, na, lb, mb, nb]
            total_weight = coeff_a * coeff_b
            T += total_weight * val * norm

    return T


def compute_kinetic_element_old(mu, nu, basis_set):
    ci, cj = basis_set[mu], basis_set[nu]
    Tij = 0.0
    for a in ci.primitives:
      for b in cj.primitives:
        zeta_a, ca, Ra, ang_a = a.zeta, a.coeff, a.center, a.angmom
        zeta_b, cb, Rb, ang_b = b.zeta, b.coeff, b.center, b.angmom
        norm = gaussian_norm(zeta_a, ang_a) * gaussian_norm(zeta_b, ang_b)

        params = compute_primitive_parameters(
          (zeta_a, Ra, ang_a),
          (zeta_b, Rb, ang_b),
          (zeta_a, Ra, ang_a),  # dummy C
          (zeta_b, Rb, ang_b)   # dummy D
        )

        I = dp_primitive_kinetic(params)

        la, ma, na = ang_a
        lb, mb, nb = ang_b

        val = I[la, ma, na, lb, mb, nb]
        Tij += ca * cb * norm * val
    return Tij


def compute_nuclear_element(mu, nu, basis_set, nuclei):
    """
    Sum over each nucleus:
     V_{μν} = Σ_A ⟨μ| -Z_A/|r−R_A| |ν⟩
    """
    ci, cj = basis_set[mu], basis_set[nu]
    Vmn = 0.0
    for a in ci.primitives:
      for b in cj.primitives:
        zeta_a, ca, Ra, la = a.zeta, a.coeff, a.center, a.angmom
        zeta_b, cb, Rb, lb = b.zeta, b.coeff, b.center, b.angmom
        norm = gaussian_norm(zeta_a, la) * gaussian_norm(zeta_b, lb)

        for (Zn, Rn) in nuclei:
            params = compute_primitive_parameters(
              (zeta_a, Ra, la),
              (zeta_b, Rb, lb),
              (zeta_a, Ra, la),  # dummy C
              (zeta_b, Rb, lb)   # dummy D
            )
            # attach P in params for nuclear recursion
            I = dp_primitive_nuclear(params, Rn, Zn)

            i,j = la[0]+lb[0], la[1]+lb[1]
            val = I[i,j,0,0,0,0]
            Vmn += ca * cb * norm * val

    return Vmn

# ——————————————————————————————————————————————
# 3) Full‐matrix builders
# ——————————————————————————————————————————————
def compute_kinetic_matrix(basis_set):
    nbf = len(basis_set)
    T = np.zeros((nbf, nbf))

    for mu in range(nbf):
        for nu in range(mu + 1):  # exploit symmetry
            val = compute_kinetic_element(mu, nu, basis_set)
            T[mu, nu] = T[nu, mu] = val

    return T


def compute_kinetic_matrix_old(basis_set):
    nbf = len(basis_set)
    T = np.zeros((nbf, nbf))
    for mu in range(nbf):
        for nu in range(mu+1):
            val = compute_kinetic_element(mu, nu, basis_set)
            T[mu, nu] = T[nu, mu] = val
    return T

def compute_nuclear_matrix(basis_set, nuclei):
    nbf = len(basis_set)
    V = np.zeros((nbf, nbf))
    for mu in range(nbf):
        for nu in range(mu+1):
            val = compute_nuclear_element(mu, nu, basis_set, nuclei)
            V[mu, nu] = V[nu, mu] = val
    return V

# ——————————————————————————————————————————————
# 4) Core Hamiltonian
# ——————————————————————————————————————————————

def compute_hcore(basis_set, nuclei):
    T = compute_kinetic_matrix(basis_set)
    V = compute_nuclear_matrix(basis_set, nuclei)
    return T + V
