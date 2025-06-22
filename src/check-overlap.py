import numpy as np

# Basis set primitives (Li 1s, Li 2s, H 1s)
Li_1s = [
    (0.7946504870, 0.4446345422),
    (2.936200663,  0.5353281423),
    (16.11957475,  0.1543289673)
]

Li_2s = [
    (0.04808867840, 0.7001154689),
    (0.1478600533,  0.3995128261),
    (0.6362897469,  -0.09996722919)
]

H_1s = [
    (0.3425250914, 0.154329),
    (0.6239137298, 0.535328),
    (1.2425670795, 0.444635)
]


# Nuclear positions (in bohr)
R_Li = np.array([0.0, 0.0, 0.0])  # Li position
R_H  = np.array([1.6, 0.0, 0.0])   # H position (R_LiH = 1.6 bohr)

# Basis function centers: [Li 1s, Li 2s, H 1s]
centers = [R_Li, R_Li, R_H]

# Compute overlap between two primitive Gaussians
def primitive_overlap(a, b, Ra, Rb):
    R_AB_sq = np.sum((Ra - Rb)**2)
    p = a + b
    return (np.pi / p)**1.5 * np.exp(-a * b * R_AB_sq / p)

# Compute overlap between two contracted basis functions
def contracted_overlap(bf1, bf2, Ra, Rb):
    S = 0.0
    for (a, da) in bf1:
        for (b, db) in bf2:
            S += da * db * primitive_overlap(a, b, Ra, Rb)
    return S

# Build the full overlap matrix (3x3)
n_bf = 3  # Li 1s, Li 2s, H 1s
S = np.zeros((n_bf, n_bf))

def normalize_bf(bf, R_center):
    """Normalize a basis function to unit overlap with itself."""
    norm = np.sqrt(contracted_overlap(bf, bf, R_center, R_center))
    return [(a, d / norm) for (a, d) in bf]

# Normalize all basis functions
Li_1s_norm = normalize_bf(Li_1s, R_Li)
Li_2s_norm = normalize_bf(Li_2s, R_Li)
H_1s_norm  = normalize_bf(H_1s, R_H)



basis_functions = [Li_1s_norm, Li_2s_norm, H_1s_norm]

for i in range(n_bf):
    for j in range(n_bf):
        S[i, j] = contracted_overlap(
            basis_functions[i], 
            basis_functions[j], 
            centers[i], 
            centers[j]
        )

print("Overlap Matrix (S):")
print(S)
