import numpy as np

def build_sto3g_basis(zeta: float) -> list[tuple[float, float]]:
    """Build STO-3G basis for a given Slater exponent (zeta)"""
    # Fundamental 1s STO-3G parameters for Zeta=1.0
    d = [0.444635, 0.535328, 0.154329]
    alpha = [0.109818, 0.405771, 2.227660]    

    basis = []
    for d, alpha in zip(d, alpha):
        alpha_scaled = alpha * (zeta ** 2)
        norm_factor = (2.0 * alpha_scaled / np.pi) ** 0.75
        d_scaled = d * norm_factor
        basis.append((alpha_scaled, d_scaled))
    
    return basis

