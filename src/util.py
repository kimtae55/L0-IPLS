import numpy as np
from sklearn.metrics import matthews_corrcoef

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

# TODO: do other metrics like precision, recall, etc
# TODO: canonical loadings visualization..? 
def evaluate_mcc(A_star, B_star, A_hat, B_hat, tol=1e-6):
    true_support_A = (np.abs(A_star) > tol).astype(int).flatten()
    est_support_A = (np.abs(A_hat) > tol).astype(int).flatten()

    true_support_B = (np.abs(B_star) > tol).astype(int).flatten()
    est_support_B = (np.abs(B_hat) > tol).astype(int).flatten()

    mcc_A = matthews_corrcoef(true_support_A, est_support_A)
    mcc_B = matthews_corrcoef(true_support_B, est_support_B)

    return mcc_A, mcc_B

def evaluate_cossim(A_star, B_star, A_hat, B_hat):
    def avg_cossim(M1, M2):
        sims = []
        for i in range(M1.shape[1]):
            a = M1[:, i]
            b = M2[:, i]
            sim = np.abs(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
            sims.append(sim)
        return np.mean(sims)

    mean_cos_A = avg_cossim(A_star, A_hat)
    mean_cos_B = avg_cossim(B_star, B_hat)

    return mean_cos_A, mean_cos_B