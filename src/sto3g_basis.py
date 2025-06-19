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


def build_sto3g_basis_2s(zeta: float, shell: str = "1s") -> list[tuple[float, float]]:
    """
    Build a contracted Gaussian basis for a given shell in the STO-3G minimal basis.
    
    Parameters:
        zeta: Slater exponent (usually atomic number Z for hydrogen-like orbitals)
        shell: Orbital shell label, e.g. '1s', '2s'
        
    Returns:
        List of (alpha, d) tuples for contracted Gaussian primitives.
    """
    if shell == "1s":
        d = [0.444635, 0.535328, 0.154329]
        alpha = [0.109818, 0.405771, 2.227660]
    elif shell == "2s":
        d = [-0.0999672, 0.399512, 0.700115]
        alpha = [0.0751386, 0.231031, 0.994203]
    else:
        raise ValueError(f"Unsupported shell: {shell}")

    basis = []
    for c, a in zip(d, alpha):
        a_scaled = a * (zeta ** 2)
        norm = (2.0 * a_scaled / np.pi) ** 0.75
        basis.append((a_scaled, c * norm))
    
    return basis

if __name__ == "__main__":

    # For HeH+ (He: Zeta=2.095, H: Zeta=1.24)
    sto3g_he = build_sto3g_basis(zeta=2.095)  # Helium basis
    sto3g_h = build_sto3g_basis(zeta=1.24)    # Hydrogen basis

    primitives_heh = [sto3g_he, sto3g_h]  # [He, H]

    # Print verification
    print("Helium STO-3G basis (Zeta=2.095):")
    for i, (alpha, d) in enumerate(sto3g_he, 1):
        print(f"Primitive {i}: α = {alpha:.8f}, d = {d:.8f}")

    print("\nHydrogen STO-3G basis (Zeta=1.24):")
    for i, (alpha, d) in enumerate(sto3g_h, 1):
        print(f"Primitive {i}: α = {alpha:.8f}, d = {d:.8f}")


