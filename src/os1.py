

def K(zeta,zetap,R,Rp):
    return 2**0.5 * (np.pi**1.25)/(zeta + zetap) np.exp(- (zeta*zetap)/(zeta+zetap)*(R-Rp)**2)

def ssss(primitives):
    k1 = K(zetaa, zetab, A, B)
    k2 = K(zetaa, zetab, A, B)
    f0 = F(0,T)
    return 1 / (zeta + eta)**0.5 * k1 * k2 * f0 

def psss()
