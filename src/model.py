import numpy as np
from sdar import sdar, asdar
import numpy as np
from tqdm import tqdm

def run_l0_ipls(X, Y, K, max_iter=100, max_ipls_iter = 100, tol=1e-6):
    """
    L0 Sparse Canonical Correlation Analysis (SCCA)

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
        Input data matrix, column centered.
    Y : np.ndarray, shape (n, q)
        Output data matrix, column centered.
    K : int
        Number of canonical vector pairs to compute.
    max_iter : int, optional
        Max iterations for inner sparse regression.
    tol : float, optional
        Convergence tolerance.
    init_method : str, optional
        Initialization method: 'svd', 'uniform', 'random', 'sparse'.

    Returns
    -------
    alpha : np.ndarray
        Canonical vectors for Y.
    beta : np.ndarray
        Canonical vectors for X.
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

        alpha, beta = init(sigma_YX=(Y.T @ X) / n,
                            sigma_X=(X.T @ X) / n,
                            sigma_Y=(Y.T @ Y) / n,
                            init_method="sparse",  # or "svd", etc.
                            n=n,
                            k=k,
                            alpha_current=np.column_stack(A_hat) if k > 0 else None,
                            beta_current=np.column_stack(B_hat) if k > 0 else None,
                            eps=1e-6)

        for _ in range(max_ipls_iter):
            alpha_prev, beta_prev = alpha.copy(), beta.copy()

            Y_tilde = Omega.T @ Y @ alpha
            beta = asdar(X, Y_tilde, max_iter_per_sdar=max_iter)
            beta /= np.sqrt(beta.T @ (X.T @ X / n) @ beta)

            X_tilde = Omega @ X @ beta
            alpha = asdar(Y, X_tilde, max_iter_per_sdar=max_iter)
            alpha /= np.sqrt(alpha.T @ (Y.T @ Y / n) @ alpha)

            if np.linalg.norm(alpha - alpha_prev) < tol and np.linalg.norm(beta - beta_prev) < tol:
                break

        A_hat.append(alpha)
        B_hat.append(beta)
        r_k = alpha.T @ (Y.T @ X / n) @ beta
        R_diag.append(r_k)

    alpha_best, beta_best = np.column_stack(A_hat), np.column_stack(B_hat)
    return alpha_best, beta_best

def init(sigma_YX, sigma_X, sigma_Y, init_method, n,
         k=0, alpha_current=None, beta_current=None, eps=1e-4, d=None):
    """
    Unified initializer for canonical vectors (L0-SCCA), returning 1D vectors.

    Parameters
    ----------
    sigma_YX : np.ndarray of shape (q, p)
        Cross-covariance matrix Y^T X / n.
    sigma_X : np.ndarray of shape (p, p)
        Covariance matrix of X (X^T X / n).
    sigma_Y : np.ndarray of shape (q, q)
        Covariance matrix of Y (Y^T Y / n).
    init_method : str
        One of 'svd', 'uniform', 'random', 'sparse'.
    n : int
        Number of samples.
    k : int, default=0
        Canonical vector index. k = 0 means first pair.
    alpha_current : np.ndarray, shape (q, k), optional
        Previously found alpha vectors.
    beta_current : np.ndarray, shape (p, k), optional
        Previously found beta vectors.
    eps : float, default=1e-4
        Threshold for support detection.
    d : int, optional
        Sparsity threshold for "sparse" init. Defaults to sqrt(n).

    Returns
    -------
    alpha : np.ndarray of shape (q,)
        Initialized alpha vector.
    beta : np.ndarray of shape (p,)
        Initialized beta vector.
    """
    q, p = sigma_YX.shape

    if init_method == "svd":
        u, _, vt = np.linalg.svd(sigma_YX, full_matrices=False)
        alpha = u[:, k]
        beta = vt[k, :]

    elif init_method == "uniform":
        alpha = np.ones(q)
        beta = np.ones(p)

    elif init_method == "random":
        alpha = np.random.randn(q)
        beta = np.random.randn(p)

    elif init_method == "sparse":
        if d is None:
            d = int(np.sqrt(n))

        if k == 0:
            thresh = np.sort(np.abs(sigma_YX.ravel()))[::-1][d]
            row_max = np.max(np.abs(sigma_YX), axis=1)
            col_max = np.max(np.abs(sigma_YX), axis=0)
            row_mask = row_max > thresh
            col_mask = col_max > thresh
            submatrix = sigma_YX[np.ix_(row_mask, col_mask)]
            u, _, vt = np.linalg.svd(submatrix, full_matrices=False)
            alpha = np.zeros(q)
            beta = np.zeros(p)
            alpha[row_mask] = u[:, 0]
            beta[col_mask] = vt[0, :]

        else:
            rho_tmp = alpha_current.T @ sigma_YX @ beta_current
            sigma_YX_tmp = sigma_YX - sigma_Y @ alpha_current @ rho_tmp @ beta_current.T @ sigma_X

            row_sum = np.sum(np.abs(alpha_current), axis=1)
            col_sum = np.sum(np.abs(beta_current), axis=1)
            id_nz_alpha = np.where(row_sum > eps)[0]
            id_nz_beta = np.where(col_sum > eps)[0]

            thresh = np.sort(np.abs(sigma_YX_tmp).ravel())[::-1][d]
            row_max = np.max(np.abs(sigma_YX_tmp), axis=1)
            col_max = np.max(np.abs(sigma_YX_tmp), axis=0)

            id_row = np.unique(np.concatenate([id_nz_alpha, np.where(row_max > thresh)[0]]))
            id_col = np.unique(np.concatenate([id_nz_beta, np.where(col_max > thresh)[0]]))

            submatrix = sigma_YX_tmp[np.ix_(id_row, id_col)]
            u, _, vt = np.linalg.svd(submatrix, full_matrices=False)

            alpha = np.zeros(q)
            beta = np.zeros(p)
            alpha[id_row] = u[:, 0]
            beta[id_col] = vt[0, :]

    else:
        raise ValueError("init_method must be one of 'svd', 'uniform', 'random', or 'sparse'")

    # Normalize to unit canonical norms
    alpha_scale = float(alpha.T @ sigma_Y @ alpha)
    beta_scale = float(beta.T @ sigma_X @ beta)
    alpha /= np.sqrt(alpha_scale)
    beta /= np.sqrt(beta_scale)

    return alpha, beta