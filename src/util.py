import numpy as np

def evaluate_subspace_error(A_star, B_star, A_hat, B_hat):
    def projection(M):
        return M @ np.linalg.inv(M.T @ M) @ M.T

    P_A_star = projection(A_star)
    P_A_hat  = projection(A_hat)
    P_B_star = projection(B_star)
    P_B_hat  = projection(B_hat)

    err_A = np.linalg.norm(P_A_star - P_A_hat, ord='fro')
    err_B = np.linalg.norm(P_B_star - P_B_hat, ord='fro')
    return err_A, err_B