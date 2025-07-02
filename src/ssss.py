import numpy as np
from math import prod, sqrt, pi, exp
from collections import defaultdict
from scipy.special import erf

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
        'zeta': zeta, 'eta': eta, 'rho': rho, 'A': A, 'B': B, 'C': C, 'D': D,
        'ssss_coeff': ssss_coeff,
        'RP_A': RP_A, 'RP_B': RP_B, 'RQ_C': RQ_C, 'RQ_D': RQ_D, 'RW_P': RW_P, 'RW_Q': RW_Q,
        'F': F, 'T': T
        }

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
    need_psss  = (la + ma + na) > 0
    need_spss  = (lb + mb + nb) > 0
    need_ssps  = (lc + mc + nc) > 0
    need_sssp  = (ld + md + nd) > 0

    need_ppss = need_psss and need_spss
    need_ppps = need_ppss and need_ssps
    need_pppp = need_ppps and need_sssp

    need_psps = need_psss and need_ssps

    # max order for (ss,ss)^(m)
    max_m = la + ma + na + lb + mb + nb + lc + mc + nc + ld + md + nd

    # Allocate full tensor
    I = [np.zeros((2,)*12) for _ in range(max_m + 1)]

    # Kronecker delta
    def delta(i, j): return 1.0 if i == j else 0.0

    # Base case up to max order [(ss,ss)^(m)]
    for m in range(max_m + 1):
        I[m][0,0,0, 0,0,0, 0,0,0, 0,0,0] = ssss_coeff * F[m]

    if need_psss:
        for m in range(max_m):
            for i in range(3):
                I[m][1 if i==0 else 0,
                     1 if i==1 else 0,
                     1 if i==2 else 0,
                     0,0,0, 0,0,0, 0,0,0] = (
                         RP_A[i] * I[m][0,0,0, 0,0,0, 0,0,0, 0,0,0] +
                         RW_P[i] * I[m+1][0,0,0, 0,0,0, 0,0,0, 0,0,0]
                         )

    if need_ppss:
        for m in range(max_m-1):
            for i in range(3):
                for j in range(3):
                    I[m][1 if i==0 else 0,
                         1 if i==1 else 0,
                         1 if i==2 else 0,
                         1 if j==0 else 0,
                         1 if j==1 else 0,
                         1 if j==2 else 0,
                         0,0,0, 0,0,0] = (
                             RP_B[j] * I[m][1 if i==0 else 0,
                                            1 if i==1 else 0,
                                            1 if i==2 else 0,
                                            0,0,0, 0,0,0, 0,0,0] +
                             RW_P[j] * I[m+1][1 if i==0 else 0,
                                              1 if i==1 else 0,
                                              1 if i==2 else 0,
                                              0,0,0, 0,0,0, 0,0,0] +
                             delta(i,j) * ( I[m][0,0,0, 0,0,0, 0,0,0, 0,0,0] -
                                            rho/zeta * I[m+1][0,0,0, 0,0,0, 0,0,0, 0,0,0])
                         )

    if need_ppps:
        for m in range(max_m-2):
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        # (p_i p_j, p_k s)^(m) =
                        # (Q_k - C_k)(p_i p_j, ss)^(m) +
                        # (W_k - Q_k)(p_i p_j, ss)^(m+1)
                        # [1/(2(zeta+eta))]*(delta(ik)(s p_j,ss)^(m+1) + delta(jk)(p_i s,ss)^(m+1) )
                        I[m][1 if i==0 else 0,
                             1 if i==1 else 0,
                             1 if i==2 else 0,
                             1 if j==0 else 0,
                             1 if j==1 else 0,
                             1 if j==2 else 0,
                             1 if k==0 else 0,
                             1 if k==1 else 0,
                             1 if k==2 else 0,
                             0,0,0] = (
                                 RQ_C[k] * I[m][1 if i==0 else 0,
                                                1 if i==1 else 0,
                                                1 if i==2 else 0,
                                                1 if j==0 else 0,
                                                1 if j==1 else 0,
                                                1 if j==2 else 0,
                                                0,0,0, 0,0,0] +
                                 RW_Q[k] * I[m+1][1 if i==0 else 0,
                                                  1 if i==1 else 0,
                                                  1 if i==2 else 0,
                                                  1 if j==0 else 0,
                                                  1 if j==1 else 0,
                                                  1 if j==2 else 0,
                                                  0,0,0, 0,0,0] +
                                 (delta(i,k) * I[m+1][0,0,0,
                                                      1 if j==0 else 0,
                                                      1 if j==1 else 0,
                                                      1 if j==2 else 0,
                                                      0,0,0, 0,0,0] +
                                  delta(j, k) * I[m+1][1 if i==0 else 0,
                                                       1 if i==1 else 0,
                                                       1 if i==2 else 0,
                                                       0,0,0, 0,0,0, 0,0,0]) / (2*(zeta+eta))
                                 ) 

    if need_pppp:
        for m in range(max_m-3):
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        # (p_i p_j, p_k s)^(m) =
                        # (Q_k - C_k)(p_i p_j, ss)^(m) +
                        # (W_k - Q_k)(p_i p_j, ss)^(m+1)
                        # [1/(2(zeta+eta))]*(delta(ik)(s p_j,ss)^(m+1) + delta(jk)(p_i s,ss)^(m+1) )
                        I[m][1 if i==0 else 0,
                             1 if i==1 else 0,
                             1 if i==2 else 0,
                             1 if j==0 else 0,
                             1 if j==1 else 0,
                             1 if j==2 else 0,
                             1 if k==0 else 0,
                             1 if k==1 else 0,
                             1 if k==2 else 0,
                             0,0,0] = (
                                 RQ_C[k] * I[m][1 if i==0 else 0,
                                                1 if i==1 else 0,
                                                1 if i==2 else 0,
                                                1 if j==0 else 0,
                                                1 if j==1 else 0,
                                                1 if j==2 else 0,
                                                0,0,0, 0,0,0] +
                                 RW_Q[k] * I[m+1][1 if i==0 else 0,
                                                  1 if i==1 else 0,
                                                  1 if i==2 else 0,
                                                  1 if j==0 else 0,
                                                  1 if j==1 else 0,
                                                  1 if j==2 else 0,
                                                0,0,0, 0,0,0] +
                                 (delta(i,k) * I[m+1][0,0,0,
                                                      1 if j==0 else 0,
                                                      1 if j==1 else 0,
                                                      1 if j==2 else 0,
                                                      0,0,0, 0,0,0] +
                                  delta(j, k) * I[m+1][1 if i==0 else 0,
                                                       1 if i==1 else 0,
                                                       1 if i==2 else 0,
                                                       0,0,0, 0,0,0, 0,0,0]) / (2*(zeta+eta))
                                 ) 


    return I


def dp_primitive_eri_oldddd(params):
    # Unpack params
    la, ma, na = params['a_ang']
    lb, mb, nb = params['b_ang']
    lc, mc, nc = params['c_ang']
    ld, md, nd = params['d_ang']
    zeta, eta, rho = params['zeta'], params['eta'], params['rho']
    A, B, C, D = params['A'], params['B'], params['C'], params['D']
    RP_A, RP_B = params['RP_A'], params['RP_B']
    RQ_C, RQ_D = params['RQ_C'], params['RQ_D']
    RW_P, RW_Q = params['RW_P'], params['RW_Q']
    ssss_coeff = params['ssss_coeff']
    F = params['F']  # Boys function values

    # Initialize 12D array for p-orbitals (indices: ia,ja,ka, ib,jb,kb, ic,jc,kc, id,jd,kd)
    I = np.zeros((2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2))  # Each component is 0 or 1

    # Helper functions
    def delta(i, j):
        return 1.0 if i == j else 0.0

    def ssss(m):
        return ssss_coeff * F[m]

    def psss(m, i):
        return RP_A[i] * ssss(m) + RW_P[i] * ssss(m + 1)

    def psps(m, i, k):
        term1 = RQ_C[k] * psss(m, i)
        term2 = RW_Q[k] * psss(m + 1, i)
        term3 = delta(i, k) / (2 * (zeta + eta)) * ssss(m + 1)

        return term1 + term2 + term3

    def ppss(m, i, j):
        term1 = RP_B[j] * psss(m, i)
        term2 = RW_P[j] * psss(m + 1, i)
        term3 = (delta(i, j) / (2 * zeta) ) * (ssss(m) - (rho / zeta) * ssss(m + 1))
        
        return term1 + term2 + term3

    def ppps(m, i, j, k):
        term1 = RQ_C[k] * ppss(m, i, j)
        term2 = RW_Q[k] * ppss(m + 1, i, j)
        term3 = ( delta(i, k) * psss(m + 1, j) + delta(j, k) * psss(m + 1, i) ) / (2 * (zeta + eta))

        return term1 + term2 + term3

    def pppp(m, i, j, k, l):
        term1 = RQ_D[l] * ppps(m, i, j, k)
        term2 = RW_Q[l] * ppps(m + 1, i, j, k)
        # term3 = ( delta(il) * spps(m, j,k) + delta(jl) * psps(m+1, i,k) ) / (2 * (zeta + eta))
        term4 = (delta(k, l) / (2 * eta)) * ( ppss(m, i, j) - (rho / eta) * ppss(m+1, i,j) )
        
        return term1 + term2 + term4

    # Base case: (ss|ss)
    I[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] = ssss(0)



   # Recursion: Only valid p-orbital combinations (exactly one '1' per particle)
    for ia, ja, ka in [(1,0,0), (0,1,0), (0,0,1)]:  # px, py, pz
        # (ps|ss)
        I[ia, ja, ka,  0, 0, 0,  0, 0, 0,  0, 0, 0] = psss(0, ia)

        for ib, jb, kb in [(1,0,0), (0,1,0), (0,0,1)]:
            # (pp|ss)
            I[ia, ja, ka,  ib, jb, kb,  0, 0, 0,  0, 0, 0] = ppss(0, ia, ib)

            for ic, jc, kc in [(1,0,0), (0,1,0), (0,0,1)]:
                # (pp|ps)
                I[ia, ja, ka,  ib, jb, kb,  ic, jc, kc,  0, 0, 0] = ppps(0, ia, ib, ic)

                for idd, jd, kd in [(1,0,0), (0,1,0), (0,0,1)]:
                    # (pp|pp)
                    I[ia, ja, ka,  ib, jb, kb,  ic, jc, kc,  idd, jd, kd] = pppp(0, ia, ib, ic, idd)

    return I



# ---- Bottom-up DP for one primitive quartet ----
def dp_primitive_eri_old(params):
    """
    params must include:
      - la,ma,na, lb,...
      - f0, f1, RP_A, RP_B, RP_C, RP_D, t, prefactor, Boys array F
    """
    # Unpack params
    la, ma, na = params['a_ang']
    lb, mb, nb = params['b_ang']
    lc, mc, nc = params['c_ang']
    ld, md, nd = params['d_ang']
    zeta = params['zeta']; eta = params['eta']
    A = params['A']; B = params['B']; C = params['C']; D = params['D']
    RP_A = params['RP_A']; RP_B = params['RP_B']
    RQ_C = params['RQ_C']; RQ_D = params['RQ_D']
    RW = params['RW']
    ssss_coeff = params['ssss_coeff']
    F = params['F']  # Boys function values up to m_max




    # Dimensions: (la+1,ma+1,na+1, lb+1,...) for A,B,C,D
    I = np.zeros((la+1, ma+1, na+1,
                  lb+1, mb+1, nb+1,
                  lc+1, mc+1, nc+1,
                  ld+1, md+1, nd+1))
    # Base: (0,0|0,0)
    I[0,0,0, 0,0,0, 0,0,0, 0,0,0] = ssss_coeff * F[0]

    # Loop over all shells in increasing "sum angular momentum"
    # You must choose an ordering so that all dependencies are computed first
    # Here: nested loops over each axis; skip the base case
    for ia in range(la+1):
      for ja in range(ma+1):
        for ka in range(na+1):
          for ib in range(lb+1):
            for jb in range(mb+1):
              for kb in range(nb+1):
                for ic in range(lc+1):
                  for jc in range(mc+1):
                    for kc in range(nc+1):
                      for idd in range(ld+1):
                        for jd in range(md+1):
                          for kd in range(nd+1):
                            if (ia,ja,ka, ib,jb,kb, ic,jc,kc, idd,jd,kd) == (0,)*12:
                                continue
                            val = 0.0
                            # lowering on Center A, x-dir
                            if ia > 0:
                                val += RP_A[0] * I[ia-1,ja,ka, ib,jb,kb, ic,jc,kc, idd,jd,kd]
                                val += RW[0] * I[ia-1,ja,ka, ib,jb,kb, ic,jc,kc, idd,jd,kd]
                                # cross-term (B)
                                if ib+1 <= lb:
                                    val += (1/(2*zeta)) * I[ia-1,ja,ka, ib+1,jb,kb, ic,jc,kc, idd,jd,kd]
                                if ia > 1:
                                    val += ((ia-1)/(2*zeta)) * I[ia-2,ja,ka, ib,jb,kb, ic,jc,kc, idd,jd,kd]

                            # lowering on Center A, y-dir
                            if ja > 0:
                                val += RP_A[1] * I[ia,ja-1,ka, ib,jb,kb, ic,jc,kc, idd,jd,kd]
                                val += RW[1] * I[ia,ja-1,ka, ib,jb,kb, ic,jc,kc, idd,jd,kd]
                                if jb+1 <= mb: # cross-term (B)
                                    val += (1/(2*zeta)) * I[ia,ja-1,ka, ib,jb+1,kb, ic,jc,kc, idd,jd,kd]
                                if ja > 1:
                                    val += ((ja-1)/(2*zeta)) * I[ia,ja-2,ka, ib,jb,kb, ic,jc,kc, idd,jd,kd]

                            # lowering on Center A, z-dir
                            if ka > 0:
                                val += RP_A[2] * I[ia,ja,ka-1, ib,jb,kb, ic,jc,kc, idd,jd,kd]
                                val += RW[2] * I[ia,ja,ka-1, ib,jb,kb, ic,jc,kc, idd,jd,kd]
                                if kb+1 <= mb: # cross-term (B)
                                    val += (1/(2*zeta)) * I[ia,ja,ka-1, ib,jb,kb+1, ic,jc,kc, idd,jd,kd]
                                if ka > 1:
                                    val += ((ka-1)/(2*zeta)) * I[ia,ja,ka-2, ib,jb,kb, ic,jc,kc, idd,jd,kd]

                            # lowering on Center B, x-dir
                            if ib > 0:
                                val += RP_B[0] * I[ia,ja,ka, ib-1,jb,kb, ic,jc,kc, idd,jd,kd]
                                val += RW[0] * I[ia,ja,ka, ib-1,jb,kb, ic,jc,kc, idd,jd,kd]
                                # cross-term (A)
                                if ia+1 <= la:
                                    val += (1/(2*zeta)) * I[ia+1,ja,ka, ib-1,jb,kb, ic,jc,kc, idd,jd,kd]
                                if ib > 1:
                                    val += ((ib-1)/(2*zeta)) * I[ia,ja,ka, ib-2,jb,kb, ic,jc,kc, idd,jd,kd]

                            # lowering on Center B, y-dir
                            if jb > 0:
                                val += RP_B[1] * I[ia,ja,ka, ib,jb-1,kb, ic,jc,kc, idd,jd,kd]
                                val += RW[1] * I[ia,ja,ka, ib,jb-1,kb, ic,jc,kc, idd,jd,kd]
                                if ja+1 <= ma: # cross-term (A)
                                    val += (1/(2*zeta)) * I[ia,ja+1,ka, ib,jb-1,kb, ic,jc,kc, idd,jd,kd]
                                if jb > 1:
                                    val += ((jb-1)/(2*zeta)) * I[ia,ja,ka, ib,jb-2,kb, ic,jc,kc, idd,jd,kd]

                            # lowering on Center B, z-dir
                            if kb > 0:
                                val += RP_B[2] * I[ia,ja,ka, ib,jb,kb-1, ic,jc,kc, idd,jd,kd]
                                val += RW[2] * I[ia,ja,ka, ib,jb,kb-1, ic,jc,kc, idd,jd,kd]
                                if ka+1 <= ma: # cross-term (A)
                                    val += (1/(2*zeta)) * I[ia,ja,ka+1, ib,jb,kb-1, ic,jc,kc, idd,jd,kd]
                                if kb > 1:
                                    val += ((kb-1)/(2*zeta)) * I[ia,ja,ka, ib,jb,kb-2, ic,jc,kc, idd,jd,kd]

                            # lowering on Center C, x-dir
                            if ic > 0:
                                val += RQ_C[0] * I[ia,ja,ka, ib,jb,kb, ic-1,jc,kc, idd,jd,kd]
                                val += RW[0] * I[ia,ja,ka, ib,jb,kb, ic-1,jc,kc, idd,jd,kd]
                                # cross-term (D)
                                if idd+1 <= ld:
                                    val += (1/(2*eta)) * I[ia,ja,ka, ib,jb,kb, ic-1,jc,kc, idd+1,jd,kd]
                                if ic > 1:
                                    val += ((ic-1)/(2*eta)) * I[ia,ja,ka, ib,jb,kb, ic-2,jc,kc, idd,jd,kd]

                            # lowering on Center C, y-dir
                            if jc > 0:
                                val += RQ_C[1] * I[ia,ja,ka, ib,jb,kb, ic,jc-1,kc, idd,jd,kd]
                                val += RW[1] * I[ia,ja,ka, ib,jb,kb, ic,jc-1,kc, idd,jd,kd]
                                if jd+1 <= md: # cross-term (D)
                                    val += (1/(2*eta)) * I[ia,ja,ka, ib,jb,kb, ic,jc-1,kc, idd,jd+1,kd]
                                if jc > 1:
                                    val += ((jc-1)/(2*eta)) * I[ia,ja,ka, ib,jb,kb, ic,jc-2,kc, idd,jd,kd]

                            # lowering on Center C, z-dir
                            if kc > 0:
                                val += RQ_C[2] * I[ia,ja,ka, ib,jb,kb, ic,jc,kc-1, idd,jd,kd]
                                val += RW[2] * I[ia,ja,ka, ib,jb,kb, ic,jc,kc-1, idd,jd,kd]
                                if kd+1 <= md: # cross-term (D)
                                    val += (1/(2*eta)) * I[ia,ja,ka, ib,jb,kb, ic,jc,kc-1, idd,jd,kd+1]
                                if kc > 1:
                                    val += ((kc-1)/(2*eta)) * I[ia,ja,ka, ib,jb,kb, ic,jc,kc-2, idd,jd,kd]

                            # lowering on Center D, x-dir
                            if idd > 0:
                                val += RQ_D[0] * I[ia,ja,ka, ib,jb,kb, ic,jc,kc, idd-1,jd,kd]
                                val += RW[0] * I[ia,ja,ka, ib,jb,kb, ic,jc,kc, idd-1,jd,kd]
                                # cross-term (C)
                                if ic+1 <= lc:
                                    val += (1/(2*eta)) * I[ia,ja,ka, ib,jb,kb, ic+1,jc,kc, idd-1,jd,kd]
                                if idd > 1:
                                    val += ((idd-1)/(2*eta)) * I[ia,ja,ka, ib,jb,kb, ic,jc,kc, idd-2,jd,kd]

                            # lowering on Center D, y-dir
                            if jd > 0:
                                val += RQ_D[1] * I[ia,ja,ka, ib,jb,kb, ic,jc,kc, idd,jd-1,kd]
                                val += RW[1] * I[ia,ja,ka, ib,jb,kb, ic,jc,kc, idd,jd-1,kd]
                                if jc+1 <= mc: # cross-term (D)
                                    val += (1/(2*eta)) * I[ia,ja,ka, ib,jb,kb, ic,jc+1,kc, idd,jd-1,kd]
                                if jd > 1:
                                    val += ((jd-1)/(2*(eta))) * I[ia,ja,ka, ib,jb,kb, ic,jc,kc, idd,jd-2,kd]

                            # lowering on Center C, z-dir
                            if kd > 0:
                                val += RQ_D[2] * I[ia,ja,ka, ib,jb,kb, ic,jc,kc, idd,jd,kd-1]
                                val += RW[2] * I[ia,ja,ka, ib,jb,kb, ic,jc,kc, idd,jd,kd-1]
                                if kc+1 <= mc: # cross-term (D)
                                    val += (1/(2*eta)) * I[ia,ja,ka, ib,jb,kb, ic,jc,kc+1, idd,jd,kd-1]
                                if kd > 1:
                                    val += ((kd-1)/(2*eta)) * I[ia,ja,ka, ib,jb,kb, ic,jc,kc, idd,jd,kd-2]


                            I[ia,ja,ka, ib,jb,kb, ic,jc,kc, idd,jd,kd] = val

    # Return the fully assembled primitive ERI
    return I[la,ma,na, lb,mb,nb, lc,mc,nc, ld,md,nd]

def unpack_key(key):
    """Convert a quartet key back into primitive parameters."""
    (zeta_a, center_a, angmom_a,
     zeta_b, center_b, angmom_b,
     zeta_c, center_c, angmom_c,
     zeta_d, center_d, angmom_d) = key
    # Convert centers back to numpy arrays if needed
    center_a = np.array(center_a)
    center_b = np.array(center_b)
    center_c = np.array(center_c)
    center_d = np.array(center_d)
    return (zeta_a, center_a, angmom_a), (zeta_b, center_b, angmom_b), \
           (zeta_c, center_c, angmom_c), (zeta_d, center_d, angmom_d)

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


def compute_eri_element_flaw(mu, nu, lam, sig, basis_set):
    """
    Contracted ERI (μ ν | λ σ):
      loops over all primitives in μ, ν, λ, σ,
      builds params for dp_primitive_eri,
      picks out the single [l,m,n,…] entry,
      multiplies by coeffs & norms, and accumulates.
    """
    # 1) grab contracted‐function metadata
    ci = basis_set[mu].primitives
    cj = basis_set[nu].primitives
    ck = basis_set[lam].primitives
    cl = basis_set[sig].primitives

    eri = 0.0

    # 2) loop over primitives
    for a_prim in ci:
      for b_prim in cj:
        for c_prim in ck:
          for d_prim in cl:

            # 3) unpack primitive data
            zeta_a, coeff_a, center_a, ang_a = a_prim
            zeta_b, coeff_b, center_b, ang_b = b_prim
            zeta_c, coeff_c, center_c, ang_c = c_prim
            zeta_d, coeff_d, center_d, ang_d = d_prim

            # 4) build the recursion parameters
            params = compute_primitive_parameters(
              (zeta_a, center_a, ang_a),
              (zeta_b, center_b, ang_b),
              (zeta_c, center_c, ang_c),
              (zeta_d, center_d, ang_d),
            )

            # 5) run Obara–Saika for *all* needed (up to p,p,p,p)
            I = dp_primitive_eri(params)

            # 6) pick off the one entry you need
            la, ma, na = ang_a
            lb, mb, nb = ang_b
            lc, mc, nc = ang_c
            ld, md, nd = ang_d
            prim_eri = I[la,ma,na, lb,mb,nb, lc,mc,nc, ld,md,nd]

            # 7) weight by contractions & normalization
            norm = ( gaussian_norm(zeta_a, ang_a)
                   * gaussian_norm(zeta_b, ang_b)
                   * gaussian_norm(zeta_c, ang_c)
                   * gaussian_norm(zeta_d, ang_d) )
            weight = coeff_a * coeff_b * coeff_c * coeff_d

            eri += weight * norm * prim_eri

    return eri


def compute_eri_element_oldd(mu, nu, lam, sig, basis_set):
    # Gather contraction primitives for each orbital
    prims_i = basis_set[mu].primitives
    prims_j = basis_set[nu].primitives
    prims_k = basis_set[lam].primitives
    prims_l = basis_set[sig].primitives

    # Map unique primitive quartets to combined weight
    quartet_map = {}
    for a in prims_i:
        for b in prims_j:
            for c in prims_k:
                for d in prims_l:
                    # Store as (zeta, coeff, center, angmom) tuples
                    key = (
                        (a.zeta, a.coeff, tuple(a.center), a.angmom),
                        (b.zeta, b.coeff, tuple(b.center), b.angmom),
                        (c.zeta, c.coeff, tuple(c.center), c.angmom),
                        (d.zeta, d.coeff, tuple(d.center), d.angmom)
                    )
                    quartet_map[key] = quartet_map.get(key, 0.0) + 1.0  # Weight handled separately

    # Loop unique quartets
    eri = 0.0
    for (a_tuple, b_tuple, c_tuple, d_tuple), _ in quartet_map.items():
        # Unpack with coefficients
        zeta_a, coeff_a, center_a, angmom_a = a_tuple
        zeta_b, coeff_b, center_b, angmom_b = b_tuple
        zeta_c, coeff_c, center_c, angmom_c = c_tuple
        zeta_d, coeff_d, center_d, angmom_d = d_tuple
        
        # Convert centers back to numpy arrays
        A = np.array(center_a)
        B = np.array(center_b)
        C = np.array(center_c)
        D = np.array(center_d)
        
        # Compute normalization
        norm = (
            gaussian_norm(zeta_a, angmom_a) * 
            gaussian_norm(zeta_b, angmom_b) * 
            gaussian_norm(zeta_c, angmom_c) * 
            gaussian_norm(zeta_d, angmom_d)
        )
        
        # Compute integral
        params = compute_primitive_parameters(
            (zeta_a, A, angmom_a),
            (zeta_b, B, angmom_b),
            (zeta_c, C, angmom_c),
            (zeta_d, D, angmom_d)
        )
        val = dp_primitive_eri(params)
        
        # Include contraction coefficients and normalization
        total_weight = coeff_a * coeff_b * coeff_c * coeff_d
        eri += total_weight * val * norm

        # print(f"Primitive key: {key}, weight: {total_weight}, ERI: {val}, weighted: {val*total_weight}")

    return eri

def compute_eri_element_old(mu, nu, lam, sig, basis_set):
    # Gather contraction primitives for each orbital
    prims_i = basis_set[mu].primitives
    prims_j = basis_set[nu].primitives
    prims_k = basis_set[lam].primitives
    prims_l = basis_set[sig].primitives

    # Map unique primitive quartets to combined weight
    quartet_map = {}
    for a in prims_i:
      for b in prims_j:
        for c in prims_k:
          for d in prims_l:
            # Create a frozenset or tuple key capturing zeta, center, angmom
            key = (a.zeta,tuple(a.center),a.angmom,
                   b.zeta,tuple(b.center),b.angmom,
                   c.zeta,tuple(c.center),c.angmom,
                   d.zeta,tuple(d.center),d.angmom)
            w = a.coeff * b.coeff * c.coeff * d.coeff
            quartet_map.setdefault(key, 0.0)
            quartet_map[key] += w

    # Loop unique quartets, compute once, multiply by weight
    eri = 0.0
    # In compute_eri_element():
    for key, weight in quartet_map.items():
        # key is (zeta_a, center_a, angmom_a, zeta_b, ...)
        zeta_a, center_a, angmom_a, \
            zeta_b, center_b, angmom_b, \
            zeta_c, center_c, angmom_c, \
            zeta_d, center_d, angmom_d = key
    
        # Compute norms directly
        norm_a = gaussian_norm(zeta_a, angmom_a)
        norm_b = gaussian_norm(zeta_b, angmom_b)
        norm_c = gaussian_norm(zeta_c, angmom_c)
        norm_d = gaussian_norm(zeta_d, angmom_d)
    
        # Reconstruct params (modify compute_primitive_parameters to accept tuples)
        params = compute_primitive_parameters(
            (zeta_a, center_a, angmom_a),
            (zeta_b, center_b, angmom_b),
            (zeta_c, center_c, angmom_c),
            (zeta_d, center_d, angmom_d)
        )
        val = dp_primitive_eri(params)
        eri += weight * val * norm_a * norm_b * norm_c * norm_d

    return eri


