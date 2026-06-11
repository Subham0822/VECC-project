import numpy as np

def tikhonov(A,b,lam=0.01):

    n = A.shape[1]

    x = np.linalg.solve(
        A.T @ A +
        lam*np.eye(n),
        A.T @ b
    )

    return x