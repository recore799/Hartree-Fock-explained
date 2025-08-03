import numpy as np
from math import prod, sqrt, pi, exp
from collections import defaultdict
from scipy.special import erf

from pyboys.boys import boys


# Index utilities
def pack_index(mu, nu):
    # Pack two indices into one unique key (assuming mu >= nu)
    return mu * (mu + 1) // 2 + nu

def unpack_index(packed_index):
    mu = int((math.sqrt(8*packed_index + 1) - 1) // 2)
    offset = mu * (mu + 1) // 2
    nu = packed_index - offset
    return mu, nu

def get_canonical_key(mu, nu, lam, sig):
    # Exploit 8-fold ERI symmetry: (ij|kl) == (ji|kl) == ...
    munu = pack_index(max(mu, nu), min(mu, nu))
    lamsig = pack_index(max(lam, sig), min(lam, sig))
    # Order the two pairs to enforce (munu >= lamsig)
    return (munu, lamsig) if munu >= lamsig else (lamsig, munu)


def angular_index(a, b, c, d):
    """
    Converts angular momentum vectors to a multi-index for tensor storage.
    """
    return tuple(a + b + c + d)  # Each a/b/c/d is a list or tuple of 3 ints


# Math functions
def double_factorial(n: int) -> int:
    """Compute double factorial n!! = n*(n-2)*(n-4)*...*1 (or 2)"""
    if n < 0:
        return 1
    return prod(range(n, 0, -2))

def boys_sequence(max_m: int, T: float) -> np.ndarray:
    F = np.zeros(max_m + 1)
    for m in range(max_m+1):
        F[m] = boys(m, T)
        
    return F

# Kronecker delta
def delta(i, j): return 1.0 if i == j else 0.0

def gaussian_norm(zeta: float, angmom: tuple[int,int,int]) -> float:
    """Compute normalization constant for a primitive Gaussian."""
    l, m, n = angmom
    prefactor = (2*zeta/np.pi)**0.75 * (4*zeta)**((l+m+n)/2)
    dfactor = np.sqrt(double_factorial(2*l-1) * double_factorial(2*m-1) * double_factorial(2*n-1))
    return prefactor * dfactor




def generate_shells_bfs(max_l):
    from collections import deque
    shells = [(0, 0, 0)]
    queue = deque(shells)
    
    while queue:
        current = queue.popleft()
        current_sum = sum(current)
        
        if current_sum >= max_l:
            continue
            
        for i in range(3):
            new_shell = list(current)
            new_shell[i] += 1
            new_shell = tuple(new_shell)
            
            if new_shell not in shells:
                shells.append(new_shell)
                queue.append(new_shell)
    
    
    return sorted(shells, key=lambda x: (sum(x), x))

# TRY ONCE THE FULL RECURSION WORKS TO TEST IMPROVEMENT
# @lru_cache(maxsize=None)
# def get_I_AC(m, a, c):
#     return I_AC.get((m, a, c), 0.0)

def build_Ia0(params):
    RP_A = params['RP_A']
    RP_Q = params['RP_Q']
    rho = params['rho']
    zeta = params['zeta']
    max_m = params['max_m']
    max_la = params['max_la']
    prefactor = params['ssss_prefactor']
    boys = params['boys_sequence']
    shells_a = params['shells_a']

    rho_over_zeta = rho / zeta
    half_zeta_inv = 0.5 / zeta

    I_A = defaultdict(float)

    # Build I^(m)(00|00)
    for m in range(max_m+1):
        I_A[(m, shells_a[0])] = prefactor * boys[m]

    # Build I(a0|00) via vertical recursion
    for a in shells_a:
        current_l = sum(a)
        n = max_m + 1 - current_l
        for m in range(n):
            for i in range(3):  # x, y, z directions
                a_plus = list(a)
                a_plus[i] += 1
                a_plus = tuple(a_plus)

                a_minus = list(a)
                a_minus[i] -= 1
                a_minus = tuple(a_minus)

                key_a = (m, a)
                key_am = (m, a_minus)
                key_am_m1 = (m+1, a_minus)

                # Recurrence relation:
                term1 = RP_A[i] * I_A.get((m, a), 0.0)
                term2 = rho_over_zeta * RP_Q[i] * I_A.get((m+1, a), 0.0)
                if a[i] > 0:
                    term3 = a[i] * half_zeta_inv * (
                        I_A.get((m, a_minus), 0.0) -
                        rho_over_zeta * I_A.get((m+1, a_minus), 0.0)
                    ) 
                else:
                    term3 = 0.0

                I_A[(m, a_plus)] = term1 - term2 + term3
    return I_A

def build_Iac(params, I_A):
    RQ_C = params['RQ_C']
    RP_Q = params['RP_Q']
    rho = params['rho']
    eta = params['eta']
    zeta = params['zeta']
    max_m = params['max_m']
    max_lc = params['max_la']
    max_la = params['max_la']
    boys = params['boys_sequence']
    shells_a = params['shells_a']
    shells_c = params['shells_c']

    rho_over_eta = rho / eta
    half_eta_inv = 0.5 / eta
    half_zeta_eta_inv = 0.5 / (zeta + eta)

    # Generate all shells (Cartesian triples) up to max_la
    shells = generate_shells_bfs(max_lc)

    I_AC = defaultdict(float)

   
    # Build I(a0|c0) via vertical recursion
    for c in shells_c:
        current_l = sum(c)
        n = max_lc - current_l
        for m in range(n):
            for a in shells_a:
                for i in range(3):  # x, y, z directions
                    c_plus = list(c)
                    c_plus[i] += 1
                    c_plus = tuple(c_plus)

                    c_minus = list(c)
                    c_minus[i] -= 1
                    c_minus = tuple(c_minus)

                    a_minus = list(a)
                    a_minus[i] -= 1
                    a_minus = tuple(a_minus)

                    key_c = (m, c)
                    key_cm = (m, c_minus)
                    key_cm_m1 = (m+1, c_minus)

                    # Recurrence relation:
                    if sum(c) == 0:
                        term1 = RQ_C[i] * I_A.get((m,a), 0.0)
                        term2 = rho_over_eta * RP_Q[i] * I_A.get((m+1, a), 0.0)
                        term3 = 0.0
                        if a[i] > 0:
                            term4 = a[i] * half_zeta_eta_inv * I_A.get((m+1, a_minus), 0.0)
                        else:
                            term4 = 0
                    else:
                        term1 = RQ_C[i] * I_AC.get((m, a, c), 0.0)
                        term2 = rho_over_eta * RP_Q[i] * I_AC.get((m+1, a, c), 0.0)
                        if c[i] > 0:
                            term3 = c[i] * half_eta_inv * (
                                I_AC.get((m, a, c_minus), 0.0) -
                                rho_over_eta * I_AC.get((m+1, a, c_minus), 0.0)
                                )
                        else:
                            term3 = 0.0
                        if a[i] > 0:
                            term4 = a[i] * half_zeta_eta_inv * I_AC.get((m+1, a_minus, c), 0.0)
                        else:
                            term4 = 0
                       


                    I_AC[(m, a, c_plus)] = term1 + term2 + term3 + term4
    return I_AC







# I tensor utilities
# set_ sets val to I[m][i] tensor element
# ret_ is for accessing said element

# Quartets
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
    I[m][0,0,0,
         1 if j==0 else 0,
         1 if j==1 else 0,
         1 if j==2 else 0,
         0,0,0, 0,0,0] = val
def ret_spss(I,m,j):
    return I[m][0,0,0,
                1 if j==0 else 0,
                1 if j==1 else 0,
                1 if j==2 else 0,
                0,0,0, 0,0,0]

# (ss|ps)^(m)
def set_ssps(I,m,k, val):
    I[m][0,0,0, 0,0,0,
         1 if k==0 else 0,
         1 if k==1 else 0,
         1 if k==2 else 0,
         0,0,0] = val
def ret_ssps(I,m,k):
    return I[m][0,0,0, 0,0,0,
                1 if k==0 else 0,
                1 if k==1 else 0,
                1 if k==2 else 0,
                0,0,0]

# (ss|sp)^(m)
def set_sssp(I,m,l, val):
    I[m][0,0,0, 0,0,0, 0,0,0,
         1 if l==0 else 0,
         1 if l==1 else 0,
         1 if l==2 else 0] = val
def ret_sssp(I,m,l):
    return I[m][0,0,0, 0,0,0, 0,0,0,
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

# (sp|sp)^(m)
def set_spsp(I,m,j,l, val):
    I[m][0,0,0,
         1 if j==0 else 0,
         1 if j==1 else 0,
         1 if j==2 else 0,
         0,0,0,
         1 if l==0 else 0,
         1 if l==1 else 0,
         1 if l==2 else 0] = val
def ret_spsp(I,m,j,l):
    return I[m][0,0,0,
                1 if j==0 else 0,
                1 if j==1 else 0,
                1 if j==2 else 0,
                0,0,0,
                1 if l==0 else 0,
                1 if l==1 else 0,
                1 if l==2 else 0]

# (ss|pp)^(m)
def set_sspp(I,m,k,l, val):
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

# (pp|sp)^(m)
def set_ppsp(I,m,i,j,l, val):
    I[m][1 if i==0 else 0,
         1 if i==1 else 0,
         1 if i==2 else 0,
         
         1 if j==0 else 0,
         1 if j==1 else 0,
         1 if j==2 else 0,
         0,0,0,
         1 if l==0 else 0,
         1 if l==1 else 0,
         1 if l==2 else 0] = val
def ret_ppsp(I,m,i,j,l):
    return I[m][1 if i==0 else 0,
                1 if i==1 else 0,
                1 if i==2 else 0,
                
                1 if j==0 else 0,
                1 if j==1 else 0,
                1 if j==2 else 0,
                0,0,0,
                1 if l==0 else 0,
                1 if l==1 else 0,
                1 if l==2 else 0]

# (ps|pp)^(m)
def set_pspp(I,m,i,k,l, val):
    I[m][1 if i==0 else 0,
         1 if i==1 else 0,
         1 if i==2 else 0,
         0,0,0,
         1 if k==0 else 0,
         1 if k==1 else 0,
         1 if k==2 else 0,

         1 if l==0 else 0,
         1 if l==1 else 0,
         1 if l==2 else 0] = val
def ret_pspp(I,m,i,k,l):
    return I[m][1 if i==0 else 0,
                1 if i==1 else 0,
                1 if i==2 else 0,
                0,0,0,
                1 if k==0 else 0,
                1 if k==1 else 0,
                1 if k==2 else 0,
                
                1 if l==0 else 0,
                1 if l==1 else 0,
                1 if l==2 else 0]

# (sp|pp)^(m)
def set_sppp(I,m,j,k,l, val):
    I[m][0,0,0,
         1 if j==0 else 0,
         1 if j==1 else 0,
         1 if j==2 else 0,

         1 if k==0 else 0,
         1 if k==1 else 0,
         1 if k==2 else 0,

         1 if l==0 else 0,
         1 if l==1 else 0,
         1 if l==2 else 0] = val
def ret_sppp(I,m,j,k,l):
    return I[m][0,0,0,
                1 if j==0 else 0,
                1 if j==1 else 0,
                1 if j==2 else 0,

                1 if k==0 else 0,
                1 if k==1 else 0,
                1 if k==2 else 0,
                
                1 if l==0 else 0,
                1 if l==1 else 0,
                1 if l==2 else 0]

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

# Doublets
# (s|s)^(m)
def set_ss(I,m, val):
    I[m][0,0,0, 0,0,0] = val
def ret_ss(I,m):
    return I[m][0,0,0, 0,0,0]

# (s|p)^(m)
def set_sp(I,m, i, val):
    I[m][0,0,0,
         1 if i==0 else 0,
         1 if i==1 else 0,
         1 if i==2 else 0] = val
def ret_sp(I,m, i):
    return I[m][0,0,0,
                1 if i==0 else 0,
                1 if i==1 else 0,
                1 if i==2 else 0]

# (p|s)^(m)
def set_ps(I,m, i, val):
    I[m][1 if i==0 else 0,
         1 if i==1 else 0,
         1 if i==2 else 0,
         0,0,0] = val
def ret_ps(I,m, i):
    return I[m][1 if i==0 else 0,
                1 if i==1 else 0,
                1 if i==2 else 0,
                0,0,0]

# (p|p)^(m)
def set_pp(I,m, i, j, val):
    I[m][1 if i==0 else 0,
         1 if i==1 else 0,
         1 if i==2 else 0,
         1 if j==0 else 0,
         1 if j==1 else 0,
         1 if j==2 else 0] = val
def ret_pp(I,m, i, j):
    return I[m][1 if i==0 else 0,
                1 if i==1 else 0,
                1 if i==2 else 0,
                1 if j==0 else 0,
                1 if j==1 else 0,
                1 if j==2 else 0]
 
# === Recursion builders ===
def build_ps(I_s, I_t, xi, RP_A):
    for i in range(3):
        # Overlap
        vals = RP_A[i] * ret_ss(I_s,0)
        set_ps(I_s, 0, i, vals)
        # Kinetic
        valt = RP_A[i] * ret_ss(I_t, 0) + 2 * xi * vals
        set_ps(I_t, 0, i, valt)
def build_psv(I_v, RP_A, RP_C):
    for m in range(2):
        for i in range(3):
            # Nuclear
            valv = RP_A[i] * ret_ss(I_v, m) - RP_C[i] * ret_ss(I_v, m+1)
            set_ps(I_v, m, i, valv)

def build_sp(I_s, I_t, xi, RP_B):
    for j in range(3):
        # Overlap
        vals = RP_B[j] * ret_ss(I_s,0)
        set_sp(I_s, 0, j, vals)
        # Kinetic
        valt = RP_B[j] * ret_ss(I_t, 0) + 2 * xi * vals
        set_sp(I_t, 0, j, valt)

def build_sp_print(I_s, I_t, xi, RP_B):
    for j in range(3):
        # Overlap
        vals = RP_B[j] * ret_ss(I_s,0)
        set_sp(I_s, 0, j, vals)
        # Kinetic
        valt = RP_B[j] * ret_ss(I_t, 0) + 2 * xi * vals
        set_sp(I_t, 0, j, valt)
        print("Val_S: ", vals)
        print("Val_T: ", valt)

def build_spv(I_v, RP_B, RP_C):
    for m in range(2):
        for j in range(3):
             # Nuclear
            valv = RP_B[j] * ret_ss(I_v, m) - RP_C[j] * ret_ss(I_v, m+1)
            set_sp(I_v, m, j, valv)
   
def build_pp(I_s, I_t, xi, zeta, RP_B):
    for i in range(3):
        for j in range(3):
            # Overlap
            vals = RP_B[j] * ret_ps(I_s, 0, i) + delta(i,j) / (2*zeta) * ret_ss(I_s, 0)
            set_pp(I_s, 0, i, j, vals)
            # Kinetic
            valt = RP_B[j] * ret_ps(I_t, 0, i) + delta(i, j) / (2*zeta) * ret_ss(I_t, 0) + 2 * xi * vals
            set_pp(I_t, 0, i, j, valt)
def build_ppv(I_v, zeta, RP_B, RP_C):
    for i in range(3):
        for j in range(3):
            # Nuclear
            valv = RP_B[j] * ret_ps(I_v, 0, i) - RP_C[j] * ret_ps(I_v, 0, i) + delta(i, j) / (2*zeta) * (ret_ss(I_v, 0) - ret_ss(I_v, 1))
            set_pp(I_v, 0, i, j, valv)


# ERI tensor
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

def build_ppss(I, RP_B, RW_P, rho, zeta, max_m):
    for m in range(max_m-1):
        for i in range(3):
            for j in range(3):
                val = (
                    RP_B[j] * ret_psss(I,m,i) +
                    RW_P[j] * ret_psss(I,m+1,i) +
                    (delta(i,j)/(2*zeta)) * (ret_ssss(I,m) - (rho/zeta) * ret_ssss(I,m+1))
                )
                set_ppss(I,m,i,j, val)

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

def build_spsp(I, RQ_D, RW_Q, zeta, eta, max_m):
    for m in range(max_m-1):
        for j in range(3):
            for l in range(3):
                val = (
                    RQ_D[l] * ret_spss(I,m,j) +
                    RW_Q[l] * ret_spss(I,m+1,j) +
                    (delta(j,l) / (2*(zeta+eta))) * ret_ssss(I,m+1)
                )
                set_spsp(I, m, j, l, val)

def build_sspp(I, RQ_D, RW_Q, rho, eta, max_m):
    for m in range(max_m - 1):
        for k in range(3):  # C center (third)
            for l in range(3):  # D center (fourth)
                val = (
                    RQ_D[l] * ret_ssps(I,m,k) +
                    RW_Q[l] * ret_ssps(I,m+1,k) +
                    delta(k,l)/(2*eta) * (ret_ssss(I,m) - (rho/eta) * ret_ssss(I,m+1))
                )
                set_sspp(I,m,k,l, val)

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
                         delta(j,k) * ret_psss(I,m+1,i)) / (2*(zeta+eta))
                    )
                    set_ppps(I,m,i,j,k, val)

def build_ppsp(I, RQ_D, RW_Q, zeta, eta, max_m):
    for m in range(max_m-2):
        for i in range(3):
            for j in range(3):
                for l in range(3):
                    val = (
                        RQ_D[l] * ret_ppss(I,m,i,j) +
                        RW_Q[l] * ret_ppss(I,m+1,i,j) +
                        (delta(i,l) * ret_spss(I,m+1,j) +
                         delta(j,l) * ret_psss(I,m+1,i)) / (2*(zeta+eta))
                    )
                    set_ppsp(I, m, i, j, l, val)

def build_pspp(I, RP_A, RW_P, zeta, eta, max_m):
    for m in range(max_m-2):
        for i in range(3):
            for k in range(3):
                for l in range(3):
                    val = (
                        RP_A[i] * ret_sspp(I, m, k, l) +
                        RW_P[i] * ret_sspp(I, m+1, k, l) +
                        (delta(i,k) * ret_sssp(I,m+1,l) +
                         delta(i,l) * ret_ssps(I, m+1, k)) / (2*(zeta+eta))
                    )

def build_sppp(I, RP_B, RW_P, zeta, eta, max_m):
    for m in range(max_m-2):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    val = (
                        RP_B[j] * ret_sspp(I, m, k, l) +
                        RW_P[j] * ret_sspp(I, m+1, k, l) +
                        (delta(j,k) * ret_sssp(I,m+1,l) +
                         delta(j,l) * ret_ssps(I, m+1, k)) / (2*(zeta+eta))
                    )

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
    F = boys_sequence(max_m, T)

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
        'zeta': zeta, 'eta': eta, 'rho': rho, 'xi': xi, 'A': A, 'B': B, 'C': C, 'D': D, 'P': P,
        'ssss_coeff': ssss_coeff,
        'RP_A': RP_A, 'RP_B': RP_B, 'RQ_C': RQ_C, 'RQ_D': RQ_D, 'RW_P': RW_P, 'RW_Q': RW_Q,
        'F': F, 'max_m': max_m, 'T': T
        }

# Recursion logic
def needs_recursion_one_electron(params):
    la, ma, na = params['a_ang']
    lb, mb, nb = params['b_ang']

    def total(lmn): return sum(lmn)

    L1 = total((la, ma, na))
    L2 = total((lb, mb, nb))

    return {
        "ps": L1 > 0,
        "sp": L2 > 0,
        "pp": L1 > 0 and L2 > 0
    }




def needs_recursion_two_electron(params):
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
    need["psps"] = need["psss"] and need["ssps"]
    need["pssp"] = need["psss"] and need["sssp"]
    need["spps"] = need["spss"] and need["ssps"]
    need["spsp"] = need["spss"] and need["sssp"]
    need["sspp"] = need["ssps"] and need["sssp"]
    need["ppps"] = need["ppss"] and need["ssps"]
    need["ppsp"] = need["ppss"] and need["sssp"]
    need["pspp"] = need["psss"] and need["sspp"]
    need["sppp"] = need["spss"] and need["sspp"]
    need["pppp"] = need["ppss"] and need["sspp"]

    return need

def primitive_st(params):
    """
    Compute S, T, and V primitive integrals.
    Returns: dict with keys 'S', 'T', 'V', each mapping to I[0][a,b] arrays
    """
    A, B = params['A'], params['B']
    RP_A, RP_B = params['RP_A'], params['RP_B']
    zeta, xi = params['zeta'], params['xi']
    P = params['P']

    RAB2 = np.dot(A - B, A - B)

    # Initialize tensors
    I_s = [np.zeros((2,)*6)]
    I_t = [np.zeros((2,)*6)]

    # === base (s|s) case ===
    # Overlap (S)
    vals = (np.pi / zeta)**1.5 * np.exp(-xi * RAB2)
    set_ss(I_s, 0, vals)
    # Kinetic (T)
    valT = xi * (3 - 2 * xi * RAB2) * vals
    set_ss(I_t, 0, valT)

    # Only do higher order terms when needed
    need = needs_recursion_one_electron(params)
    if need["ps"]: build_ps(I_s, I_t, xi, RP_A) # Builds s and t recursions
    if need["sp"]: build_sp(I_s, I_t, xi, RP_B)
    if need["pp"]: build_pp(I_s, I_t, xi, zeta, RP_B)

    return I_s[0], I_t[0]

def primitive_v(params):
    A, B = params['A'], params['B']
    RP_A, RP_B = params['RP_A'], params['RP_B']
    zeta, xi = params['zeta'], params['xi']
    max_m = params['max_m']
    P = params['P']
    R_nuc = params['R_nuc']
    RP_nuc = P - R_nuc

    RAB2 = np.dot(A - B, A - B)
    U = zeta * np.dot(RP_nuc, RP_nuc)
    F = boys_sequence(max_m, U)
    I_v = [np.zeros((2,)*6) for _ in range(max_m+1)]

    vals = (np.pi / zeta)**1.5 * np.exp(-xi * RAB2)
    # Nuclear (V)
    for m in range(max_m+1):
        val = 2*(zeta / pi)**0.5 * vals * F[m]
        set_ss(I_v, m, val)

    # Only do higher order terms when needed
    need = needs_recursion_one_electron(params)
    if need["ps"]: build_psv(I_v, RP_A, RP_nuc) # Builds v recursion
    if need["sp"]: build_spv(I_v, RP_B, RP_nuc)
    if need["pp"]: build_ppv(I_v, zeta, RP_B, RP_nuc)

    return I_v[0]

def primitive_eri(params):
    # Unpack parameters
    la, ma, na = params['a_ang']
    lb, mb, nb = params['b_ang']
    lc, mc, nc = params['c_ang']
    ld, md, nd = params['d_ang']
    zeta, eta, rho = params['zeta'], params['eta'], params['rho']
    RP_A, RP_B = params['RP_A'], params['RP_B']
    RQ_C, RQ_D = params['RQ_C'], params['RQ_D']
    RW_P, RW_Q = params['RW_P'], params['RW_Q']
    F, max_m = params['F'], params['max_m']
    ssss_coeff = params['ssss_coeff']

    # Determine whether each vertical recursion layer is needed
    need = needs_recursion_two_electron(params)

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
    if need["psps"]:  build_psps(I, RQ_C, RW_Q, zeta, eta, max_m)
    if need["spsp"]:  build_spsp(I, RQ_D, RW_Q, zeta, eta, max_m)
    if need["spps"]:  build_spps(I, RQ_C, RW_Q, zeta, eta, max_m)
    if need["pssp"]:  build_pssp(I, RQ_D, RW_Q, zeta, eta, max_m)
    if need["sspp"]:  build_sspp(I, RQ_D, RW_Q, rho, eta, max_m)

    if need["ppps"]:  build_ppps(I, RQ_C, RW_Q, zeta, eta, max_m)
    if need["ppsp"]:  build_ppsp(I, RQ_D, RW_Q, zeta, eta, max_m)
    if need["pspp"]:  build_pspp(I, RP_A, RW_P, zeta, eta, max_m)
    if need["sppp"]:  build_sppp(I, RP_B, RW_P, zeta, eta, max_m)

    if need["pppp"]:  build_pppp(I, RQ_D, RW_Q, zeta, eta, rho, max_m)


    return I


def compute_one_electron_element(mu, nu, basis_set, nuclei):
    """
    Compute the one-electron integrals S_{μν}, T_{μν}, V_{μν} using unified OS recursion.
    Returns:
        (S_mu_nu, T_mu_nu, V_mu_nu)
    """
    ci = basis_set[mu]
    cj = basis_set[nu]

    S = 0.0
    T = 0.0
    V = 0.0

    for i, a_prim in enumerate(ci.primitives):
        for j, b_prim in enumerate(cj.primitives):
            zeta_a, coeff_a, center_a, ang_a = a_prim.zeta, a_prim.coeff, a_prim.center, a_prim.angmom
            zeta_b, coeff_b, center_b, ang_b = b_prim.zeta, b_prim.coeff, b_prim.center, b_prim.angmom

            norm = (
                gaussian_norm(zeta_a, ang_a) *
                gaussian_norm(zeta_b, ang_b)
            )

            coeff = coeff_a * coeff_b * norm

            params = compute_primitive_parameters(
                (zeta_a, center_a, ang_a),
                (zeta_b, center_b, ang_b),
                (zeta_a, center_a, ang_a),
                (zeta_b, center_b, ang_b)
                )

            la, ma, na = ang_a
            lb, mb, nb = ang_b

            # if la == 0 and ma == 0 and na == 0 and nb == 1:
            #     print(f"mu={mu}, nu={nu}, ang_a={ang_a}, ang_b={ang_b}, needs: {needs_recursion_one_electron(params)}")

            prim_S, prim_T = primitive_st(params)
                
            # if la == 0 and ma == 0 and na == 0 and nb == 1:
            #     print("Prim_S: ", prim_S[la, ma, na, lb, mb, nb])
            #     print("Prim_T: ", prim_T[la, ma, na, lb, mb, nb])

            S += coeff * prim_S[la, ma, na, lb, mb, nb]
            T += coeff * prim_T[la, ma, na, lb, mb, nb]


            # Nuclear attraction is looped over all nuclei
            for Z_nuc, R_nuc in nuclei:
                # Add nuclear center to params for computing prim_V
                params["R_nuc"] = np.array(R_nuc)
               

                prim_V = primitive_v(params)

                V += -Z_nuc * coeff * prim_V[la, ma, na, lb, mb, nb]

    return S, T, V


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
                    val_tensor = primitive_eri(params)

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

   
def build_one_electron_matrices(basis_set, nuclei):
    nbf = len(basis_set)
    S = np.zeros((nbf, nbf))
    T = np.zeros((nbf, nbf))
    V = np.zeros((nbf, nbf))
    H = np.zeros((nbf, nbf))

    for mu in range(nbf):
        for nu in range(mu+1):
            s, t, v = compute_one_electron_element(mu, nu, basis_set, nuclei)
            S[mu, nu] = S[nu, mu] = s
            T[mu, nu] = T[nu, mu] = t
            V[mu, nu] = V[nu, mu] = v

    H = T + V
    return S, T, V, H


