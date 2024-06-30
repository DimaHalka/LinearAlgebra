import numpy as np


def svd(M):
    eigenvalues, eigenvectors = np.linalg.eig(np.dot(M, M.T))
    ncols = np.argsort(eigenvalues)[::-1]
    U = eigenvectors[:, ncols]

    eigenvalues, eigenvectors = np.linalg.eig(np.dot(M.T, M))
    ncols = np.argsort(eigenvalues)[::-1]
    VT = eigenvectors[:, ncols].T

    if (np.size(np.dot(M, M.T)) > np.size(np.dot(M.T, M))):
        newM = np.dot(M.T, M)
    else:
        newM = np.dot(M, M.T)
    eigenvalues, eigenvectors = np.linalg.eig(newM)
    eigenvalues = np.sqrt(eigenvalues)
    sigma = eigenvalues[::-1]

    return U, sigma, VT


A = np.array([[4,2,0],
              [1,5,6]])
U, sigma, VT = svd(A)


new_sigma = np.zeros((2, 3))
new_sigma[:2, :2] = np.diag(sigma[:2])

print(A,"\n")

A_remake = (U @ new_sigma @ VT)
print(A_remake)