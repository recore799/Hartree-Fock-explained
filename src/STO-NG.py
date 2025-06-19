import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad

def slater_orbital(r, zeta=1.0):
    """1s Slater-type orbital"""
    return np.exp(-zeta * r) * np.sqrt(zeta**3 / np.pi)

def gaussian_orbital(r, alpha):
    """1s Gaussian-type orbital"""
    return (2 * alpha / np.pi)**(3/4) * np.exp(-alpha * r**2)

def contracted_gaussian(r, alphas, coeffs):
    """Contracted Gaussian function (STO-NG)"""
    return sum(d * gaussian_orbital(r, a) for a, d in zip(alphas, coeffs))

def overlap_integral(params, zeta=1.0, N=3):
    """
    Calculate overlap between Slater and contracted Gaussian.
    params contains both exponents and coefficients.
    """
    alphas = params[:N]
    coeffs = params[N:]
    
    # Normalize the contracted Gaussian
    norm = quad(lambda r: 4*np.pi*r**2 * contracted_gaussian(r, alphas, coeffs)**2, 0, np.inf)[0]
    norm = np.sqrt(norm)
    normalized_cgf = lambda r: contracted_gaussian(r, alphas, coeffs) / norm
    
    # Calculate overlap
    integrand = lambda r: 4*np.pi*r**2 * slater_orbital(r, zeta) * normalized_cgf(r)
    S = quad(integrand, 0, np.inf)[0]
    
    # We minimize -S to maximize S
    return -S

def get_initial_guess(N, zeta):
    """Provide better initial guesses based on known values"""
    if N == 1:
        return np.array([0.3*zeta]), np.array([1.0])
    elif N == 2:
        return np.array([0.15*zeta, 0.9*zeta]), np.array([0.7, 0.4])
    elif N == 3:
        return np.array([0.1*zeta, 0.4*zeta, 2.2*zeta]), np.array([0.45, 0.55, 0.15])
    else:
        # For N>3, use geometric progression
        alpha_min = 0.1*zeta
        alpha_max = 5.0*zeta
        alphas = alpha_min * (alpha_max/alpha_min)**(np.arange(N)/(N-1))
        return alphas, np.ones(N)/N

def optimize_sequential(zeta=1.0, N=3, max_iter=20, tol=1e-6):
    """Sequential optimization of exponents and coefficients"""
    # Initial guess
    # In general these guesses should help find a local minimum, but I couldn't get it to find the same as Szabo, so I built a function based on known values
    # alphas = np.linspace(0.01, 1.0, N)
    # coeffs = np.ones(N)
    alphas, coeffs = get_initial_guess(N, zeta)

   
    prev_overlap = 0
    for i in range(max_iter):
        # Optimize exponents with fixed coefficients
        def overlap_alphas(alpha_params):
            return overlap_integral(np.concatenate([alpha_params, coeffs]), zeta, N)
        
        res_alpha = minimize(
            overlap_alphas,
            alphas,
            bounds=[(1e-6, None)]*N,
            method='L-BFGS-B'
        )
        alphas = res_alpha.x
        
        # Optimize coefficients with fixed exponents
        def overlap_coeffs(coeff_params):
            return overlap_integral(np.concatenate([alphas, coeff_params]), zeta, N)
        
        res_coeff = minimize(
            overlap_coeffs,
            coeffs,
            bounds=[(None, None)]*N,
            method='L-BFGS-B'
        )
        coeffs = res_coeff.x
        
        current_overlap = -res_coeff.fun
        if abs(current_overlap - prev_overlap) < tol:
            break
        prev_overlap = current_overlap
    
    return alphas, coeffs, current_overlap


if __name__ == "__main__":

    # Optimize STO-3G for Zeta=1.0
    alphas, coeffs, max_overlap = optimize_sequential(zeta=1.0, N=3, max_iter=100)
    print(f"STO-3G exponents: {alphas}")
    print(f"STO-3G coefficients: {coeffs}")
    print(f"Maximum overlap: {max_overlap}")

    # For H2 with Zeta=1.24
    # alphas_h2, coeffs_h2, overlap_h2 = optimize_sequential(zeta=1.24, N=3)
    # For different values of zeta I can just do alpha * zeta**2
    alphas_h2 = alphas * 1.24**2
    print(f"H2 STO-3G exponents: {alphas_h2}")
    print(f"H2 STO-3G coefficients: {coeffs}")
    print(f"H2 Maximum overlap: {max_overlap}")

