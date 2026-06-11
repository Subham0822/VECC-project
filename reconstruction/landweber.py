import numpy as np

def landweber(A,b,
              iterations=100,
              alpha=0.001):

    x = np.zeros(A.shape[1])

    for _ in range(iterations):

        x = x + alpha * (
            A.T @ (b - A @ x)
        )

    return x