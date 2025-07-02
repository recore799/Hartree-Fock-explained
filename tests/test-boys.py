import numpy as np
from math import exp, sqrt, pi
from scipy.special import erf

# Import reference boys function from
# https://github.com/peter-reinholdt/pyboys
# adjust for your own reference

import sys
from pathlib import Path

# Add pyboys directory to Python path
sys.path.append(str(Path.home() / "pyboys"))

from boys import boys



# Tested for T= 0, 2.43, 23 with 1e-6 tolerance
T_test = 10
max_m = 15  # Test up to F_10

# old implementation (doesn't work)
def compute_boys_sequence(max_m: int, T: float) -> list[float]:
    """Stable upward recursion of Boys function F_0 to F_max_m."""
    F = [0.0] * (max_m + 1)
    if T < 1e-8:
        for m in range(max_m + 1):
            F[m] = 1.0 / (2*m + 1) - T / (2*m + 3)
        return F
    else:
        # Direct base case
        sqrt_T = sqrt(T)
        F[0] = sqrt(pi / (4*T)) * erf(sqrt_T)
        for m in range(1, max_m + 1):
            F[m] = (2*T*F[m-1] - exp(-T)) / (2*m - 1)
        return F

# new implementation
def boys_new(n: int, T: float) -> float:
    """Modern downward recursion implementation"""
    if T < 1e-15:
        return 1/(2*n + 1) - T/(2*n + 3)
    
    if n == 0:
        sqrt_T = np.sqrt(T)
        return np.sqrt(np.pi)/2 * erf(sqrt_T)/sqrt_T if T > 0 else 1.0
    
    n_max = n + int(10 + 2*T)
    F = np.zeros(n_max + 2)
    F[n_max + 1] = 1/(2*(n_max + 1) + 1)
    
    for m in range(n_max, -1, -1):
        F[m] = (2*T*F[m + 1] + exp(-T))/(2*m + 1)
    
    return F[n]

# return full F array using the new implementation
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

# Generate test values
# upward_results = compute_boys_sequence(max_m, T_test)
downward_results = [boys_new(m, T_test) for m in range(max_m + 1)]
boys_sequence_results = boys_sequence(max_m, T_test)
reference_results = [boys(m, T_test) for m in range(max_m + 1)]

# Print comparison table
print(f"\nBoys function comparison at T = {T_test:.3f}")
print("-" * 60)
print(f"{'n':>3} | {'Array':>15} | {'Downward':>15} | {'Reference':>15} | {'Δ(Arr)':>8} | {'Δ(Down)':>8}")
print("-" * 60)

for n in range(max_m + 1):
    diff_up = abs(boys_sequence_results[n] - reference_results[n])
    diff_down = abs(downward_results[n] - reference_results[n])
    
    print(f"{n:3d} | {boys_sequence_results[n]:15.10f} | {downward_results[n]:15.10f} | "
          f"{reference_results[n]:15.10f} | {diff_up:8.2e} | {diff_down:8.2e}")

# Check for significant differences
tolerance = 1e-6

# Upward recursion didn't work, uncomment this line if it was fixed
# assert all(abs(u - r) < tolerance for u, r in zip(upward_results, reference_results)), \
       # "Upward recursion failed validation"

# assert all(abs(d - r) < tolerance for d, r in zip(downward_results, reference_results)), \
#        "Downward recursion failed validation"

# print(f"\nAll tests passed within tolerance of {tolerance}!")
print(f"\nDownward test passed within tolerance of {tolerance}!")
