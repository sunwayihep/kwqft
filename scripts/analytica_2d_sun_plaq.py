import numpy as np
from scipy.special import iv

def D_nu(N, nu, x):
    M = np.empty((N, N))
    for i in range(N):
        for j in range(N):
            M[i, j] = iv(nu + i - j, x)
    return np.linalg.det(M)


def D_nu_prime(N, nu, x):
    M = np.empty((N, N))
    Mp = np.empty((N, N))

    for i in range(N):
        for j in range(N):
            n = nu + i - j
            M[i, j] = iv(n, x)

            # I_n'(x) = [I_{n-1}(x) + I_{n+1}(x)] / 2
            Mp[i, j] = 0.5 * (iv(n-1, x) + iv(n+1, x))

    # d det(M)/dx = det(M) Tr(M^{-1} M')
    return np.linalg.det(M) * np.trace(np.linalg.solve(M, Mp))


def plaquette(N, beta, nu_max=30):
    x = beta / N

    Z = D_nu(N, 0, x)
    Zp = D_nu_prime(N, 0, x)

    for nu in range(1, nu_max + 1):
        Z += 2.0 * D_nu(N, nu, x)
        Zp += 2.0 * D_nu_prime(N, nu, x)

    return Zp / (N * Z)

if __name__ == "__main__":
    for beta in range(1, 20):
        for N in [2, 3, 4]:
            print(f"beta={beta}, N={N}, plaq={plaquette(N, beta=beta)}")
        print()
