import numpy as np
import time

from collections import defaultdict

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

def generate_shells_bfs_np(max_l):
    from collections import deque
    shells = [np.array([0, 0, 0])]
    queue = deque(shells)
    
    while queue:
        current = queue.popleft()
        current_sum = np.sum(current)
        
        if current_sum >= max_l:
            continue
            
        for i in range(3):
            new_shell = current.copy()
            new_shell[i] += 1
            
            if not any(np.array_equal(new_shell, s) for s in shells):
                shells.append(new_shell)
                queue.append(new_shell)
    
    # Convert to a single NumPy array and sort
    shells_array = np.array(shells)
    shells_array = shells_array[np.lexsort((shells_array[:, 2], shells_array[:, 1], shells_array[:, 0]))]
    shells_array = shells_array[np.argsort(np.sum(shells_array, axis=1))]
    
    return shells_array

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

def build_Ia0_new(params):
    RP_A = params['RP_A']
    RP_Q = params['RP_Q']
    rho = params['rho']
    zeta = params['zeta']
    max_m = params['max_m']
    max_la = params['max_la']
    prefactor = params['ssss_prefactor']
    boys = params['boys_sequence']
    shells_a = params['shells_a']
    one = params['one'] # 3X3 identity matrix

    rho_over_zeta = rho / zeta
    half_zeta_inv = 0.5 / zeta

    I_A = defaultdict(float)

    # Build I^(m)(00|00)
    for m in range(max_m+1):
        I_A[(m, (0,0,0))] = prefactor * boys[m]

    # Build I(a0|00) via vertical recursion
    for a in shells_a:
        a_np = np.array(a)
        current_l = np.sum(a)
        n = max_la + 1 - current_l

        # Precompute raising and lowering keys
        a_plus = a + one
        a_minus = a - one

        # Convert to tuples for dict keys
        a_plus_tuples = [tuple(vec) for vec in a_plus]
        a_minus_tuples = [tuple(vec) for vec in a_minus]

        for m in range(n):

            # Vectorized base terms
            term1 = RP_A * I_A.get((m,a), 0.0)
            term2 = RP_Q * rho_over_zeta * I_A.get((m+1, a), 0.0)

            term = term1 - term2

            for i, key in enumerate(a_plus_tuples):
                if a[i] > 0:
                    term3 = a[i] * half_zeta_inv * (
                        I_A.get((m, a_minus_tuples[i]), 0.0)
                        - rho_over_zeta * I_A.get((m+1, a_minus_tuples[i]))
                        )
                    I_A[(m, key)] = term[i] + term3
                else:
                    I_A[(m, key)] = term[i]

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


# one_i = np.zeros((3,3,3), dtype=int)
# np.einsum('iii->i', one_i)[:] = 1

# shells_a = generate_shells_bfs_np(2)

# some_dict = defaultdict(int)
# f = [1.0,0.5,2.0]
# some_1darray = np.array(f)
# m = 0

# for a in shells_a:
#     a_plus = a + one_i  # Shape: (3, 3, 3)
    
#     # Extract the diagonal (i,i,i) elements -> shape (3, 3)
#     a_plus_diag = np.einsum('iii->i', a_plus)  # Equivalent to [a_plus[0,0], a_plus[1,1], a_plus[2,2]]
    
#     # Convert to tuples of integers
#     a_plus_tuples = [tuple(map(int, vec)) for vec in a_plus_diag]
    
#     # Vectorized dictionary update
#     for a_plus_tuple in a_plus_tuples:
#         print(a_plus_tuples)
#         x = some_1darray * 3
#         print(x)
#         some_dict[(m, a_plus_tuple)] = some_1darray * some_dict.get((m, tuple(a)), 0.0)

# for k, v in some_dict.items():
#     print(f"I^{k[0]}({k[1]}) = {v:.5f}")

# Initialize one_i correctly (3x3x3 with 1 on diagonal)
# start_time1 = time.time()
# one_i = np.zeros((3, 3, 3), dtype=int)
# np.einsum('iii->i', one_i)[:] = 1

# # Generate test shells (e.g., up to l=2)
# shells_a = np.array([
#     [0, 0, 0],
#     [1, 0, 0],
#     [0, 1, 0],
#     [0, 0, 1]
# ])

# some_dict = defaultdict(float)
# some_dict[(0, (0,0,0))] = 1
# f = [1.0, 0.5, 2.0]
# some_1darray = np.array(f)
# m = 0
# for i in range(100):
#     for a in shells_a:
#         # Get all 3 possible a_plus vectors (x+1, y+1, z+1)
#         a_plus = a + one_i  # Shape: (3, 3, 3)
    
#         # Extract the diagonal vectors (where i=j=k)
#         a_plus_vectors = np.array([a_plus[i, i, :] for i in range(3)])  # Shape: (3, 3)
    
#         # Convert to integer tuples
#         a_plus_tuples = [tuple(vec.astype(int)) for vec in a_plus_vectors]
    
#         # Update dictionary
#         for i, a_plus_tuple in enumerate(a_plus_tuples):
#             some_dict[(m, a_plus_tuple)] = some_1darray[i] * some_dict.get((m, tuple(a)), 0.0)
# end_time1 = time.time()

# # Print results
# for k, v in some_dict.items():
#     print(f"I^{k[0]}{k[1]} = {v:.5f}")

# start_time2 = time.time()
# one_i = np.zeros((3, 3, 3), dtype=int)
# np.einsum('iii->i', one_i)[:] = 1  # Diagonal unit vectors

# shells_a = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])  # Example shells
# some_dict = defaultdict(float)
# some_dict[(0, (0, 0, 0))] = 1.0  # Initial condition
# some_1darray = np.array([1.0, 0.5, 2.0])  # Example factors
# m = 0

# for i in range(100):
#     for a in shells_a:
#         a_plus = a + one_i  # Shape: (3, 3, 3)
#         a_plus_vectors = np.array([a_plus[i, i] for i in range(3)])  # Get x/y/z increments
#         a_plus_tuples = [tuple(vec.astype(int)) for vec in a_plus_vectors]
    
#         # Vectorized computation
#         current_val = some_dict.get((m, tuple(a)), 0.0)
#         new_values = some_1darray * current_val  # Vectorized multiply
    
#         # Minimal loop for dict assignment
#         for i, a_plus_tuple in enumerate(a_plus_tuples):
#             some_dict[(m, a_plus_tuple)] = new_values[i]

# end_time2 = time.time()



# # Print results
# for k, v in some_dict.items():
#     print(f"new I^{k[0]}{k[1]} = {v:.5f}")



shells_a_np = generate_shells_bfs_np(4)
shells_a = generate_shells_bfs(4)
# shells_a = generate_shells_bfs(5)
# shells_c = generate_shells_bfs(5)

params_np = {
    'RQ_C': np.array([0.1, 0.2, 0.3]),
    'RP_A': np.array([0.1, -0.2, 0.3]),
    'RP_Q': np.array([0.4, 0.5, 0.6]),
    'rho': 1.2,
    'zeta': 0.9,
    'eta': 0.7,
    'max_m': 3,
    'max_la': 3,
    'max_lc': 5,
    'ssss_prefactor': 1.0,
    'boys_sequence': [1.0, 0.5, 0.333, 0.25, 0.20, 1.66],
    'shells_a': generate_shells_bfs(4),
    'one': np.eye(3, dtype=int)
    # 'shells_c': shells_c
}


params = {
    'RQ_C': [0.1, 0.2, 0.3],
    'RP_A': [0.1, -0.2, 0.3],
    'RP_Q': [0.4, 0.5, 0.6],
    'rho': 1.2,
    'zeta': 0.9,
    'eta': 0.7,
    'max_m': 3,
    'max_la': 3,
    'max_lc': 5,
    'ssss_prefactor': 1.0,
    'boys_sequence': [1.0, 0.5, 0.333, 0.25, 0.20, 1.66],
    'shells_a': generate_shells_bfs(4),
    'one': np.eye(3, dtype=int)
    # 'shells_c': shells_c
}


# start_time = time.time()
# I_A_np = build_Ia0_new(params_np)
# I_A = build_Ia0(params)
# I_AC = build_Iac_vectorized(params, I_A)
# end_time = time.time()



# start_time1 = time.time()
# I_A = build_Ia0(params)
# I_AC = build_Iac(params, I_A)
# end_time1 = time.time()

# print the nonzero keys
# for k, v in I_A.items():
#     print(f"np I^{k[0]}({k[1]}|0) = {v:.5f}")

# for k, v in I_A_np.items():
#     print(f"I^{k[0]}({k[1]}|0) = {v:.5f}")





# def original_method():
#     shells_a = generate_shells_bfs_np(4)
#     one_i = np.zeros((3, 3, 3), dtype=int)
#     np.einsum('iii->i', one_i)[:] = 1
#     some_dict = defaultdict(float)
#     some_dict[(0, (0,0,0))] = 1
#     some_1darray = np.array([1.0, 0.5, 2.0])
#     m = 0
#     for _ in range(100):
#         for a in shells_a:
#             a_plus = a + one_i
#             a_plus_vectors = np.array([a_plus[i, i, :] for i in range(3)])
#             a_plus_tuples = [tuple(vec.astype(int)) for vec in a_plus_vectors]
#             for i, a_plus_tuple in enumerate(a_plus_tuples):
#                 some_dict[(m, a_plus_tuple)] = some_1darray[i] * some_dict.get((m, tuple(a)), 0.0)
#     return some_dict

# def optimized_method():
#     shells_a = generate_shells_bfs_np(4)
#     one_i = np.zeros((3, 3, 3), dtype=int)
#     np.einsum('iii->i', one_i)[:] = 1
#     some_dict = defaultdict(float)
#     some_dict[(0, (0,0,0))] = 1.0
#     some_1darray = np.array([1.0, 0.5, 2.0])
#     m = 0
#     for _ in range(100):
#         for a in shells_a:
#             a_plus = a + one_i
#             a_plus_vectors = np.array([a_plus[i, i] for i in range(3)])
#             a_plus_tuples = [tuple(vec.astype(int)) for vec in a_plus_vectors]
#             current_val = some_dict.get((m, tuple(a)), 0.0)
#             new_values = some_1darray * current_val
#             for i, a_plus_tuple in enumerate(a_plus_tuples):
#                 some_dict[(m, a_plus_tuple)] = new_values[i]
#     return some_dict

# def optimized_method1():
#     shells_a = generate_shells_bfs_np(4)
#     one = np.eye(3, dtype=int)  # Simpler than 3D array
#     some_dict = defaultdict(float)
#     some_dict[(0, (0,0,0))] = 1.0
#     some_1darray = np.array([1.0, 0.5, 2.0])
#     m = 0

#     for _ in range(100):
#         for a in shells_a:
#             a_plus = a + one  # Shape: (3, 3)
#             a_plus_tuples = [tuple(vec) for vec in a_plus]  # No need for astype(int) since one is already int
#             current_val = some_dict.get((m, tuple(a)), 0.0)
#             new_values = some_1darray * current_val  # Vectorized multiply
            
#             for i, a_plus_tuple in enumerate(a_plus_tuples):
#                 some_dict[(m, a_plus_tuple)] = new_values[i]
    
#     print(a_plus_tuples)
#     return some_dict


# aaaa = optimized_method1()

import timeit

# n_repeats = 100
# time_original = timeit.timeit(original_method, number=n_repeats)
# time_optimized = timeit.timeit(optimized_method, number=n_repeats)

# print(f"Original: {time_original/n_repeats:.6f} sec per call")
# print(f"Optimized: {time_optimized/n_repeats:.6f} sec per call")
# print(f"Speedup: {time_original/time_optimized:.2f}x")

def test_original():
    I_A = build_Ia0(params)
    return I_A

def test_optimized():
    I_A = build_Ia0_new(params_np)
    return I_A

n_repeats = 10
time_original = timeit.timeit(test_original, number=n_repeats)
time_optimized = timeit.timeit(test_optimized, number=n_repeats)

# 4. Print results
print(f"Original: {time_original/n_repeats:.6f} sec per call")
print(f"Optimized: {time_optimized/n_repeats:.6f} sec per call")
print(f"Speedup: {time_original/time_optimized:.2f}x")

# 5. Verify results match
# result_orig = test_original()
# result_opt = test_optimized()
# assert result_orig == result_opt, "Results differ!"
# # Ensure both methods produce identical results
# dict_orig = original_method()
# dict_opt = optimized_method()
# assert dict_orig == dict_opt, "Results differ!"


# print(f"Total runtime1: {end_time1 - start_time1:.4f} seconds")


# print(f"Total runtime2: {end_time2 - start_time2:.4f} seconds")

