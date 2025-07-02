def pack_index(a, b):
    # Pack two indices into one unique key (assuming a >= b)
    return a * (a + 1) // 2 + b

def get_canonical_key(a, b, c, d):
    # Exploit 8-fold symmetry
    ab = pack_index(max(a, b), min(a, b))
    cd = pack_index(max(c, d), min(c, d))
    return (ab, cd) if ab >= cd else (cd, ab)
