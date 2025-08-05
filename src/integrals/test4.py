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
    max_la = params['max_la']
    max_lb = params['max_lb']
    max_lc = params['max_lc']
    max_ld = params['max_ld']
    prefactor = params['ssss_prefactor']
    boys = params['boys_sequence']
    shells_a = params['shells_a']
    n_shells = params['n_shells']

    rho_over_zeta = rho / zeta
    half_zeta_inv = 0.5 / zeta

    c = (0,0,0)
    I_A = defaultdict(float)

    # Build I^(m)(00|00)
    n1 = max_la + max_lb + max_lc + max_ld + 1
    for m in range(n1):
        I_A[(m, c, c)] = prefactor * boys[m] # Introduce the c key early

    # Build I^(m)(a0|00) via vertical recursion
    stop = n_shells[max_la+max_lb]
    for a in shells_a[:stop]:
        current_l = sum(a)
        n = max_la + max_lb + max_lc + max_ld - current_l
        for m in range(n): # Determines how deep the recursion goes
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
                term1 = RP_A[i] * I_A.get((m, a, c), 0.0)
                term2 = rho_over_zeta * RP_Q[i] * I_A.get((m+1, a, c), 0.0)
                if a[i] > 0:
                    term3 = a[i] * half_zeta_inv * (
                        I_A.get((m, a_minus), 0.0) -
                        rho_over_zeta * I_A.get((m+1, a_minus, c), 0.0)
                    ) 
                else:
                    term3 = 0.0

                I_A[(m, a_plus, c)] = term1 - term2 + term3
    return I_A

def build_Iac(params, I_A):
    RQ_C = params['RQ_C']
    RP_Q = params['RP_Q']
    rho = params['rho']
    eta = params['eta']
    zeta = params['zeta']
    max_la = params['max_la']
    max_lb = params['max_lb']
    max_lc = params['max_lc']
    max_ld = params['max_ld']
    boys = params['boys_sequence']
    shells_a = params['shells_a']
    shells_c = params['shells_c']
    n_shells = params['n_shells']

    rho_over_eta = rho / eta
    half_eta_inv = 0.5 / eta
    half_zeta_eta_inv = 0.5 / (zeta + eta)

    I_AC = defaultdict(float)

    # Port I^(m)(00|00) to I_AC
    n1 = max_la + max_lb + max_lc + max_ld + 1
    for m in range(n1):
        I_AC[(m,(0,0,0),(0,0,0))] = I_A.get((m,(0,0,0), (0,0,0)),0.0)

    # Build I(a0|c0) via vertical recursion
    # Generate p shell from I_A first
    for a in shells_a:
        for m in range(max_lc+max_ld):
            for i in range(3):
                c_plus = [0,0,0]
                c_plus[i] += 1
                c_plus = tuple(c_plus)

                a_minus = list(a)
                a_minus[i] -= 1
                a_minus = tuple(a_minus)

                term1 = RQ_C[i] * I_A.get((m,a, (0,0,0)), 0.0)
                term2 = rho_over_eta * RP_Q[i] * I_A.get((m+1, a, (0,0,0)), 0.0)
                if a[i] > 0:
                    term4 = a[i] * half_zeta_eta_inv * I_A.get((m+1, a_minus, (0,0,0)), 0.0)
                else:
                    term4 = 0
                I_AC[(m, a, c_plus)] = term1 + term2 + term4

    # Generete higher shells only when needed 
    stop = n_shells[max_lc + max_ld]
    if max_lc > 1 or max_ld > 0:
        for c in shells_c[1:stop]:
            current_l = sum(c)
            n2 = max_lc + max_ld - current_l
            for m in range(n2):
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



def build_Iabc(params, I_AC):
    A = params['A']
    B = params['B']
    max_la = params['max_la']
    max_lb = params['max_lb']
    max_lc = params['max_lc']
    max_ld = params['max_ld']
    n_shells = params['n_shells']
    shells_a = params['shells_a']
    shells_b = params['shells_b']
    shells_c = params['shells_c']

    AB = A - B

    d = (0,0,0)

    # Initialize tensor
    I_ABC = defaultdict(float)

    # Select shells from max_lax+max_lb - 1 to max_la (in reverse)
    relevant_shells_a = get_relevant_shells(shells_a, max_la, max_la + max_lb - 1)

    start = n_shells[max_lc]
    for c in shells_c[start:]: # SHOULD WORK
        for a in relevant_shells_a: # Loop in reversed order
            for i in range(3):
                b_plus = [0,0,0]
                b_plus[i] += 1
                b_plus = tuple(b_plus)
                
                a_plus = list(a)
                a_plus[i] += 1
                a_plus = tuple(a)

                term1 = I_AC.get((0, a_plus, c), 0.0)
                # print("term1: ", term1)
                term2 = AB[i] * I_AC.get((0, a, c), 0.0)
                # print("term2: ", term2)

                I_ABC[(a, b_plus, c, d)] = term1 + term2

    stop1 = n_shells[max_lb]
    if max_lb >= 1:
        for c in shells_c[start:]: # SHOULD WORK
            for a in relevant_shells_a: # Loop in reversed order
                for b in shells_b[1:stop1]:
                    for i in range(3):
                        b_plus = list(b)
                        b_plus[i] += 1
                        b_plus = tuple(b_plus)
                    
                        a_plus = list(a)
                        a_plus[i] += 1
                        a_plus = tuple(a)

               
                        term1 = I_ABC.get((a_plus, b, c), 0.0)
                        term2 = AB[i] * I_ABC.get((a, b, c), 0.0)

                        I_ABC[(a, b_plus, c, d)] = term1 + term2

    return I_ABC

def build_Iabcd(params, I_ABC):
    C = params['C']
    D = params['D']
    max_la = params['max_la']
    max_lb = params['max_lb']
    max_lc = params['max_lc']
    max_ld = params['max_ld']
    n_shells = params['n_shells']
    shells_a = params['shells_a']
    shells_b = params['shells_b']
    shells_c = params['shells_c']

    CD = C - D

    # Initialize tensor
    I_ABCD = defaultdict(float)

    # Select shells from max_lax+max_lb - 1 to max_la (in reverse)
    relevant_shells_c = get_relevant_shells(shells_c, max_lc, max_lc + max_ld - 1)

    # Might be missing I(ab|c0) in I_ABCD
    start1 = n_shells[max_la]
    end1 = n_shells[max_la + 1]
    start2 = n_shells[max_lb]
    for a in shells_a[start1:end1]: # Loop in reversed order
        for b in shells_b[start2:]:
            for c in relevant_shells_c: # SHOULD WORK
                for i in range(3):
                    d_plus = [0,0,0]
                    d_plus[i] += 1
                    d_plus = tuple(d_plus)
                
                    c_plus = list(a)
                    c_plus[i] += 1
                    c_plus = tuple(a)

                    term1 = I_ABC.get((a, b, c_plus), 0.0)
                    term2 = CD[i] * I_AC.get((a, b, c), 0.0)

                    I_ABCD[(a, b, c, d_plus)] = term1 + term2

    if max_ld > 1:
        for a in shells_a[start1:end1]: 
            for b in shells_b[start2:]:
                for c in relevant_shells_c: # SHOULD WORK
                    for d in shells_d[1:]:
                        for i in range(3):
                            d_plus = list(d)
                            d_plus[i] += 1
                            d_plus = tuple(d_plus)
                    
                            c_plus = list(c)
                            c_plus[i] += 1
                            c_plus = tuple(c)

               
                            term1 = I_ABCD.get((a, b, c_plus, d), 0.0)
                            term2 = CD[i] * I_ABCD.get((a, b, c, d), 0.0)

                            I_ABCD[(a, b, c, d_plus)] = term1 + term2

    return I_ABCD


def get_relevant_shells(shells, min_sum, max_sum):
    """Returns shells where min_sum <= sum(shell) <= max_sum, in reverse order."""
    start_idx = next(i for i, shell in enumerate(shells) if sum(shell) >= min_sum)
    end_idx = next(
        len(shells) - 1 - i 
        for i, shell in enumerate(reversed(shells)) 
        if sum(shell) <= max_sum
    )
    return list(reversed(shells[start_idx : end_idx + 1]))


def generate_shells_bfs(max_l):
    from collections import deque
    shells = [(0, 0, 0)]
    if max_l == 0:
        return shells
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
    1: 1,    # s + p: (0,0,0), (1,0,0), (0,1,0), (0,0,1)
    2: 4,   # s + p + d: (0,0,0), ..., (2,0,0), ..., (0,0,2)
    3: 10,   # s + p + d + f
    4: 20,   # s + p + d + f + g
    5: 35,
    6: 56,
    7: 84,
    8: 120,
    9: 165,
    10: 220,
}


benchmark= {
    'max_la': 1,
    'max_lb': 1,
    'max_lc': 0,
    'max_ld': 0,
    'shells_a': 2, # How many shells to generate max_la + max_lb
    'shells_b': 1,  
    'shells_c': 0, # max_lc + max_ld
    'shells_d': 0, # max_ld - 1

}

shells_a = generate_shells_bfs(benchmark['shells_a'])
shells_b = generate_shells_bfs(benchmark['shells_b'])
shells_c = generate_shells_bfs(benchmark['shells_c'])
shells_d = generate_shells_bfs(benchmark['shells_d'])

params = {
    'A': np.array([1.0, 1.0, 1.0]),
    'B': np.array([-1.0, 1.0, 1.0]),
    'C': np.array([1.0, 1.0, 1.0]),
    'D': np.array([-1.0, 1.0, 1.0]),
    'RQ_C': [0.1, 0.2, 0.3],
    'RP_A': [0.1, -0.2, 0.3],
    'RP_Q': [0.4, 0.5, 0.6],
    'rho': 1.2,
    'zeta': 0.9,
    'eta': 0.7,
    'max_la': benchmark['max_la'],
    'max_lc': benchmark['max_lc'],
    'max_lb': benchmark['max_lb'],
    'max_ld': benchmark['max_ld'],
    'ssss_prefactor': 1.0,
    'boys_sequence': [1.0, 0.5, 0.333, 0.25, 0.20, 1.66, 1.0, 1.0, 1.0,1.0,1.0, 10.0, 10.0],
    'shells_a': shells_a,
    'shells_b': shells_b,
    'shells_c': shells_c,
    'shells_d': shells_d,
    'n_shells': N_SHELLS
}
# # --- Simple test ---
# start_time = time.time()
# I_A = build_Ia0(params)
# I_AC = build_Iac(params, I_A)
# end_time = time.time()

# # Print results I_A
# print("--- Printing I_A --- \n")
# for k, v in I_A.items():
#     print(f"I^{k[0]}({k[1]}|0) = {v:.5f}")


# # Print results I_AC
# print("--- Printing I_AC --- \n")
# for k, v in I_AC.items():
#     print(f"I^{k[0]}({k[1]}|{k[2]}) = {v:.2f}")

# print(f"Total runtime2: {end_time - start_time:.8f} seconds")

# Test relevant shells

# g = generate_shells_bfs(4)

# new_g = get_relevant_shells(g, 1, 3)

# print(list(new_g))

# --- Simple test I_ABC ---
start_time = time.time()
I_A = build_Ia0(params)
# I_AC = build_Iac(params, I_A)
I_ABC = build_Iabc(params, I_A)
# I_ABCD = build_Iabcd(params, I_ABC)
end_time = time.time()

# Print results I_A
print("--- Printing I_A --- \n")
for k, v in I_A.items():
    print(f"I^{k[0]}({k[1]}|0) = {v:.3f}")

# Print results I_AC
# print("--- Printing I_AC --- \n")
# for k, v in I_AC.items():
#     print(f"I^{k[0]}({k[1]}|{k[2]}) = {v:.8f}")

# Print results I_ABC
print("--- Printing I_ABC --- \n")
for k, v in I_ABC.items():
    print(f"I({k[0]}{k[1]}|{k[2]}{k[3]}) = {v:.3f}")


# Print results I_ABCD
# print("--- Printing I_ABCD --- \n")
# for k, v in I_ABCD.items():
#     print(f"I({k[0]}{k[1]}|{k[2]}{k[3]}) = {v:.5f}")


print(f"Total runtime2: {end_time - start_time:.8f} seconds")


# # --- Benchmark Functions ---
# def run_new_code():
#     I_A = build_Ia0(params)
#     I_AC = build_Iac(params, I_A)
#     return I_AC

# def run_old_code():
#     I_A = build_Ia0_old(params)
#     I_AC = build_Iac_old(params, I_A)
#     return I_AC

# # --- Timing ---
# new_time = timeit(run_new_code, number=1000) / 100  # Average over 100 runs
# old_time = timeit(run_old_code, number=1000) / 100

# # --- Results ---
# print("\n--- Benchmark Parameters ---")
# print(benchmark)
# print("\n--- Benchmark Results 1000 runs ---")
# print(f"Old code: {old_time:.6f} sec/call")
# print(f"New code: {new_time:.6f} sec/call")
# print(f"Speedup: {old_time / new_time:.2f}x")
# print(f"Time saved: {(old_time - new_time)*1000:.2f} ms per call")
