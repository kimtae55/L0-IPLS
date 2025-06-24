import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef

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

def evaluate_support_metrics(A_star, B_star, A_hat, B_hat, tol=1e-6):
    """
    Evaluate support recovery metrics for sparse CCA solutions.

    Parameters
    ----------
    A_star : ndarray of shape (q, K)
        Ground truth canonical vectors for Y (sparse).
    B_star : ndarray of shape (p, K)
        Ground truth canonical vectors for X (sparse).
    A_hat : ndarray of shape (q, K)
        Estimated canonical vectors for Y.
    B_hat : ndarray of shape (p, K)
        Estimated canonical vectors for X.
    tol : float, optional
        Threshold for determining nonzero entries (default is 1e-6).

    Returns
    -------
    metrics : dict
        Dictionary containing the following support recovery metrics for A and B:
        - 'sensitivity_A', 'specificity_A', 'precision_A', 'f1_A', 'mcc_A'
        - 'sensitivity_B', 'specificity_B', 'precision_B', 'f1_B', 'mcc_B'

        These metrics are computed based on binary support (nonzero) patterns
        between the ground truth and estimated canonical vectors.
    """
    def support_metrics(true, est):
        true_bin = (np.abs(true) > tol).astype(int).flatten()
        est_bin = (np.abs(est) > tol).astype(int).flatten()

        TP = np.sum((true_bin == 1) & (est_bin == 1))
        TN = np.sum((true_bin == 0) & (est_bin == 0))
        FP = np.sum((true_bin == 0) & (est_bin == 1))
        FN = np.sum((true_bin == 1) & (est_bin == 0))

        sens = TP / (TP + FN + 1e-10)
        spec = TN / (TN + FP + 1e-10)
        prec = precision_score(true_bin, est_bin, zero_division=0)
        f1 = f1_score(true_bin, est_bin, zero_division=0)
        mcc = matthews_corrcoef(true_bin, est_bin)

        return sens, spec, prec, f1, mcc

    sens_A, spec_A, prec_A, f1_A, mcc_A = support_metrics(A_star, A_hat)
    sens_B, spec_B, prec_B, f1_B, mcc_B = support_metrics(B_star, B_hat)

    return {
        'sensitivity_A': sens_A, 'specificity_A': spec_A, 'precision_A': prec_A, 'f1_A': f1_A, 'mcc_A': mcc_A,
        'sensitivity_B': sens_B, 'specificity_B': spec_B, 'precision_B': prec_B, 'f1_B': f1_B, 'mcc_B': mcc_B
    }

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

def evaluate_canonical_correlation(A_hat, B_hat, Sigma_YY, Sigma_XX, Sigma_YX):
    rho_list = []
    for k in range(A_hat.shape[1]):
        a = A_hat[:, k]
        b = B_hat[:, k]
        num = a.T @ Sigma_YX @ b
        denom = np.sqrt(a.T @ Sigma_YY @ a) * np.sqrt(b.T @ Sigma_XX @ b)
        rho = np.abs(num / (denom + 1e-10))
        rho_list.append(rho)
    return rho_list