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


def build_Ia0_wrong(params):
    RP_A = params['RP_A']
    RP_Q = params['RP_Q']
    rho = params['rho']
    zeta = params['zeta']
    max_m = params['max_m']
    max_la = params['max_la']
    prefactor = params['ssss_prefactor']
    boys = params['boys_sequence']

    rho_over_zeta = rho / zeta
    half_zeta_inv = 0.5 / zeta

    # Generate all shells (Cartesian triples) up to max_la
    shells = generate_shells_bfs(max_la)

    I_A = defaultdict(float)

    # Build I^(m)(00|00)
    for m in range(max_m+1):
        I_A[(m, shells[0])] = prefactor * boys[m]

    # Build I(a0|00) via recursion
    for a in shells:
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
                if a_minus[i] > 0:
                    term3 = a_minus[i] * half_zeta_inv * (
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



shells_a = generate_shells_bfs(5)
shells_c = generate_shells_bfs(5)

params = {
    'RQ_C': [0.1, 0.2, 0.3],
    'RP_A': [0.1, -0.2, 0.3],
    'RP_Q': [0.4, 0.5, 0.6],
    'rho': 1.2,
    'zeta': 0.9,
    'eta': 0.7,
    'max_m': 5,
    'max_la': 5,
    'ssss_prefactor': 1.0,
    'boys_sequence': [1.0, 0.5, 0.333, 0.25, 0.20, 1.66],
    'shells_a': shells_a,
    'shells_c': shells_c
}
I_A = build_Ia0(params)
I_AC = build_Iac(params, I_A)

# print the nonzero keys
for k, v in I_AC.items():
    print(f"I^{k[0]}({k[1]}|{k[2]}) = {v:.5f}")
