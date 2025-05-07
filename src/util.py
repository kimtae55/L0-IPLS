import numpy as np

def evaluate_subspace_error(A_star, B_star, A_hat, B_hat):
    def projection(M):
        Q, _ = np.linalg.qr(M)
        return Q @ Q.T
    err_A = np.linalg.norm(projection(A_star) - projection(A_hat), ord='fro')
    err_B = np.linalg.norm(projection(B_star) - projection(B_hat), ord='fro')
    return err_A, err_B