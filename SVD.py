import numpy as np


def svd(M):
    eigvals_U, eigvecs_U = np.linalg.eig(np.dot(M, M.T))
    sorted_indices_U = np.argsort(eigvals_U)[::-1]
    U = eigvecs_U[:, sorted_indices_U]
    sigma = np.sqrt(np.abs(eigvals_U[sorted_indices_U]))

    V = []
    for i in range(len(sigma)):
        if sigma[i] > 1e-10:  # avoid zero devision
            V.append(np.dot(M.T, U[:, i]) / sigma[i])
        else:
            V.append(np.zeros_like(M.T[:, 0]))
    V = np.array(V).T

    Sigma = np.zeros((U.shape[0], V.shape[1]))
    for i in range(min(U.shape[0], V.shape[1])):
        Sigma[i, i] = sigma[i]

    return U, Sigma, V.T


A = np.array([[4, 2, 9],
              [1, 5, 6]])

U, Sigma, VT = svd(A)

print("Original matrix A:")
print(A, "\n")

A_remake = U @ Sigma @ VT
print("Reconstructed matrix A:")
print(A_remake)
