
# For now, only hydrogen (Z = 1)
sto3g_data = {
    1: {  # Hydrogen
        "exponents": [3.42525091, 0.62391373, 0.16885540],
        "coefficients": [0.15432897, 0.53532814, 0.44463454]
    },
}

def build_STO3G_basis(atom_coords, atom_numbers):
    basis = []
    for coord, Z in zip(atom_coords, atom_numbers):
        data = sto3g_data[Z]
        basis.append({
            "center": coord,
            "exponents": data["exponents"],
            "coefficients": data["coefficients"]
        })
    return basis

# Define molecule
atom_coords = [
    [0.0, 0.0, 0.0],  # Atom 1 at origin
    [0.0, 0.0, 1.4]   # Atom 2 at 1.4 bohr along z-axis
]
atom_numbers = [1, 1]  # Both are hydrogen

# Build the basis
basis_functions = build_STO3G_basis(atom_coords, atom_numbers)

print(basis_functions)
