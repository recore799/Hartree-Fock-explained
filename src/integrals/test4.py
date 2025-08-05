import numpy as np
from collections import defaultdict
from timeit import timeit
import time

# result = np.zeros((3, 3, 3))
# np.einsum('iii->i', result)[:] = 1  # Set diagonal elements to 1

# one = np.eye(3, dtype=int)
# a = (1,1,1)
# a = np.array(a)

# n = a + one

# print(n)

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
    # n_shells = params['n_shells']

    rho_over_zeta = rho / zeta
    half_zeta_inv = 0.5 / zeta

    I_A = defaultdict(float)

    # Build I^(m)(00|00)
    for m in range(max_m):
        I_A[(m, shells_a[0])] = prefactor * boys[m]

    # Build I(a0|00) via vertical recursion
    for a in shells_a:
        current_l = sum(a)
        n = max_m - current_l
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
    max_lc = params['max_lc']
    max_la = params['max_la']
    boys = params['boys_sequence']
    shells_a = params['shells_a']
    shells_c = params['shells_c']
    n_shells = params['n_shells']

    rho_over_eta = rho / eta
    half_eta_inv = 0.5 / eta
    half_zeta_eta_inv = 0.5 / (zeta + eta)

    I_AC = defaultdict(float)

    # Port I(00|00) to I_AC
    I_AC[(0,(0,0,0),(0,0,0))] = I_A.get((0,(0,0,0)),0.0)

    n_shells_to_process_a = n_shells[max_la]

    # Build I(a0|c0) via vertical recursion
    # Generate p shell from I_A first
    for a in shells_a[:n_shells_to_process_a]:
        for i in range(3):
            c_plus = [0,0,0]
            c_plus[i] += 1
            c_plus = tuple(c_plus)

            a_minus = list(a)
            a_minus[i] -= 1
            a_minus = tuple(a_minus)

            term1 = RQ_C[i] * I_A.get((0,a), 0.0)
            term2 = rho_over_eta * RP_Q[i] * I_A.get((1, a), 0.0)
            if a[i] > 0:
                term4 = a[i] * half_zeta_eta_inv * I_A.get((1, a_minus), 0.0)
            else:
                term4 = 0
            I_AC[(0, a, c_plus)] = term1 + term2 + term4

    # Generete higher shells only when needed 
    if max_lc >= 2:
        for c in shells_c:
            current_l = sum(c)
            n = max_lc - current_l
            for m in range(n):
                for a in shells_a[:n_shells_to_process_a]:
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


def build_Ia0_old(params):
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
        if sum(a) == max_la:
            break
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

def build_Iac_old(params, I_A):
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

    I_AC = defaultdict(float)

    # Port I(00|00) to I_AC
    I_AC[(0, (0,0,0), (0,0,0))] = I_A.get((0, (0,0,0)), 0.0)

    # Build I(a0|c0) via vertical recursion
    for c in shells_c:
        if sum(c) == max_lc:
            break
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


# --- Test Setup ---
# Index of last tuple in shell? s=0, p=1, d=2, f=3, g=4
N_SHELLS = {
    0: 0,   # s: (0,0,0)
    1: 4,    # s + p: (0,0,0), (1,0,0), (0,1,0), (0,0,1)
    2: 10,   # s + p + d: (0,0,0), ..., (2,0,0), ..., (0,0,2)
    3: 20,   # s + p + d + f
    4: 35,   # s + p + d + f + g
    5: 100,
}



benchmark= {
    'max_la': 4,
    'max_lc': 1,
    'max_m': 4,
    'shells_a': 8, # How many shells to generate
    'shells_c': 8,

}

shells_a = generate_shells_bfs(benchmark['shells_a'])
shells_c = generate_shells_bfs(benchmark['shells_c'])

params = {
    'RQ_C': [0.1, 0.2, 0.3],
    'RP_A': [0.1, -0.2, 0.3],
    'RP_Q': [0.4, 0.5, 0.6],
    'rho': 1.2,
    'zeta': 0.9,
    'eta': 0.7,
    'max_m': benchmark['max_m'],
    'max_la': benchmark['max_la'],
    'max_lc': benchmark['max_lc'],
    'ssss_prefactor': 1.0,
    'boys_sequence': [1.0, 0.5, 0.333, 0.25, 0.20, 1.66, 1.0, 1.0, 1.0,1.0,1.0],
    'shells_a': shells_a,
    'shells_c': shells_c,
    'n_shells': N_SHELLS
}

# --- Simple test ---
# start_time = time.time()
# I_A = build_Ia0(params)
# I_AC = build_Iac(params, I_A)
# end_time = time.time()


# # Print results I_AC
# for k, v in I_AC.items():
#     print(f"new I^{k[0]}({k[1]}|{k[2]}) = {v:.2f}")

# print(f"Total runtime2: {end_time - start_time:.8f} seconds")


# Print results I_A
# for k, v in I_A.items():
#     print(f"new I^{k[0]}({k[1]}|0) = {v:.5f}")


# --- Benchmark Functions ---
def run_new_code():
    I_A = build_Ia0(params)
    I_AC = build_Iac(params, I_A)
    return I_AC

def run_old_code():
    I_A = build_Ia0_old(params)
    I_AC = build_Iac_old(params, I_A)
    return I_AC

# --- Timing ---
new_time = timeit(run_new_code, number=1000) / 100  # Average over 100 runs
old_time = timeit(run_old_code, number=1000) / 100

# --- Results ---
print("\n--- Benchmark Parameters ---")
print(benchmark)
print("\n--- Benchmark Results 1000 runs ---")
print(f"Old code: {old_time:.6f} sec/call")
print(f"New code: {new_time:.6f} sec/call")
print(f"Speedup: {old_time / new_time:.2f}x")
print(f"Time saved: {(old_time - new_time)*1000:.2f} ms per call")
