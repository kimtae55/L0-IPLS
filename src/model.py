import numpy as np
from sdar import sdar, asdar

def l0_scca(X, Y, K, lambda_alpha_list, lambda_beta_list, mode="sdar",
            tau=None, L=None, max_iter=100, tol=1e-6):
    """
    SDAR/ASDAR-based L0-penalized Sparse CCA.

    Parameters:
        X, Y: centered data matrices (n x p), (n x q)
        K: number of canonical vectors
        lambda_alpha_list, lambda_beta_list: list of regularization parameters
        mode: 'sdar' or 'asdar'
        tau, L: required if mode == 'asdar'
        max_iter: max inner iterations
        tol: convergence tolerance

    Returns:
        A_hat: (q x K), canonical vectors for Y
        B_hat: (p x K), canonical vectors for X
    """
    n, p = X.shape
    _, q = Y.shape
    A_hat, B_hat, R_diag = [], [], []
    Omega = np.eye(n)

    for k in range(K):
        if k > 0:
            A_stack = np.column_stack(A_hat)
            B_stack = np.column_stack(B_hat)
            R_mat = np.diag(R_diag)
            Omega = np.eye(n) - (Y @ A_stack @ R_mat @ B_stack.T @ X.T) / n

        alpha = np.random.randn(q)
        beta = np.random.randn(p)
        alpha /= np.sqrt(alpha.T @ (Y.T @ Y / n) @ alpha)
        beta  /= np.sqrt(beta.T @ (X.T @ X / n) @ beta)

        for _ in range(max_iter):
            alpha_prev, beta_prev = alpha.copy(), beta.copy()

            Y_tilde = Omega.T @ Y @ alpha
            if mode == "sdar":
                beta = sdar(X, Y_tilde, lambda_beta_list[k], tol=tol)
            elif mode == "asdar":
                beta = asdar(X, Y_tilde, tau=tau, L=L, beta0=beta, tol=tol)[0]
            else:
                raise ValueError("mode must be 'sdar' or 'asdar'")
            beta /= np.sqrt(beta.T @ (X.T @ X / n) @ beta)

            X_tilde = Omega @ X @ beta
            if mode == "sdar":
                alpha = sdar(Y, X_tilde, lambda_alpha_list[k], tol=tol)
            elif mode == "asdar":
                alpha = asdar(Y, X_tilde, tau=tau, L=L, beta0=alpha, tol=tol)[0]
            alpha /= np.sqrt(alpha.T @ (Y.T @ Y / n) @ alpha)

            if np.linalg.norm(alpha - alpha_prev) < tol and np.linalg.norm(beta - beta_prev) < tol:
                break

        A_hat.append(alpha)
        B_hat.append(beta)
        r_k = alpha.T @ (Y.T @ X / n) @ beta
        R_diag.append(r_k)

    return np.column_stack(A_hat), np.column_stack(B_hat)