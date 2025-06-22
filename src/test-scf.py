import numpy as np
from integrals import build_integral_arrays
from sto3g_basis import build_sto3g_basis_2s

if __name__ == "__main__":
    primitives_lithium = [
        build_sto3g_basis_2s(6.3, shell="1s"),
        build_sto3g_basis_2s(1.0, shell="2s"),
    ]

    primitives_hydrogen = [
        build_sto3g_basis_2s(1.0, shell="1s"),
    ]
    # Li_1s = [
    #     (4.35867642, 0.206812), 
    #     (16.10505099, 0.093430), 
    #     (88.4158254, 3.171430)
    # ]
    # Li_2s = [
    #     (0.536092, -0.119332),  # Note: Negative coefficient!
    #     (0.981245, -0.160854), 
    #     (2.240585, 1.143456)
    # ]
    # H_1s = [
    #     (0.3425250914, 0.154329), 
    #     (0.6239137298, 0.535328), 
    #     (1.2425670795, 0.444635)
    # ]

    Li_1s = [
        (0.7946504870, 0.4446345422),
        (2.936200663, 0.5353281423),
        (16.11957475, 0.1543289673),
        ]

    Li_2s = [
        (0.04808867840, 0.7001154689),
        (0.1478600533, 0.3995128261),
        (0.6362897469, -0.09996722919),
        ]

    H_1s = [
        (1.2425670795, 0.444635454),
        (0.6239137298, 0.53532814), 
        (3.425250914, 0.15432897),
    ]


    def norm_prim(primitives, zeta=1.0):

        basis = []
        for alpha, d in primitives:
            alpha_scaled = alpha * (zeta ** 2)
            norm_factor = (2.0 * alpha_scaled / np.pi) ** 0.75
            d_scaled = d * norm_factor
            basis.append((alpha_scaled, d_scaled))
    
        return basis

    Li_1s_norm = norm_prim(Li_1s)
    Li_2s_norm = norm_prim(Li_2s)
    H_1s_norm = norm_prim(H_1s)

    primitives_lih = [Li_1s_norm, Li_2s_norm, H_1s_norm]

    pos_lih = np.array([[0,0,0],[0,0,0],[1.6,0,0]])
    # pos = [
    #     np.array([0.0]),  # Li
    #     np.array([0.0]),  # Li again (for 2s)
    #     np.array([1.6]),  # H
    # ]
    
    Z_nuc = [3,1]
    R_nuc = np.array([[0.0, 0.0, 0.0], [1.6, 0.0, 0.0]])
    Z_lih = (3.0, 3.0, 1.0)  # match order of basis centers
    n_elec_lih = 4


    S, H_core, eri_dict = build_integral_arrays(primitives=primitives_lih, pos=pos_lih, Z=Z_lih)

    # print("Primitives: ", primitives_lih)
    print("S matrix:\n", S)


