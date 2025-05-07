import numpy as np

def sdar(X, y, T, beta0=None, max_iter=100, tol=1e-8):
    """
    SDAR algorithm for L0-penalized sparse regression.

    Parameters:
        X: (n, p) data matrix
        y: (n,) response vector
        T: sparsity level (support size)
        beta0: (p,) initial guess (default: zeros)
        max_iter: maximum number of iterations
        tol: support change tolerance

    Returns:
        beta_hat: estimated sparse coefficients
        support: active set indices
    """
    n, p = X.shape
    S = np.arange(p)
    if beta0 is None:
        beta = np.zeros(p)
    else:
        beta = beta0.copy()

    r = y - X @ beta
    d = X.T @ r / n

    for k in range(max_iter):
        pd = beta + d
        _T = int(T)
        top_indices = np.argsort(-np.abs(pd))[:_T]
        support = np.zeros(p, dtype=bool)
        support[top_indices] = True
        A_k = support
        I_k = ~A_k

        # Check support convergence
        if k > 0 and np.array_equal(A_k, prev_A_k):
            break
        prev_A_k = A_k.copy()

        # Zero out inactive parts
        beta[I_k] = 0
        d[A_k] = 0

        # Least squares on active set
        XA = X[:, A_k]
        beta_A = np.linalg.pinv(XA.T @ XA) @ XA.T @ y
        beta[A_k] = beta_A

        # Dual update
        r = y - X @ beta
        d[I_k] = X[:, I_k].T @ r / n

    return beta, np.where(A_k)[0]

def asdar(X, y, tau, L, beta0=None, max_outer=100, max_inner=100, tol=1e-8, stop_criterion=None):
    """
    Adaptive SDAR (ASDAR) wrapper over SDAR.

    Parameters:
        X: (n, p) data matrix
        y: (n,) response vector
        tau: step size controlling support growth (e.g., tau=1,2,...)
        L: maximum support size
        beta0: initial guess (default: zeros)
        max_outer: max outer iterations for ASDAR
        max_inner: max inner iterations passed to SDAR
        tol: tolerance for support convergence in SDAR
        stop_criterion: optional callable: f(beta_k, d_k, k) -> bool

    Returns:
        beta_hat: estimated coefficients
        support: indices of final active set
    """
    n, p = X.shape
    if beta0 is None:
        beta = np.zeros(p)
    else:
        beta = beta0.copy()

    # initialize dual variable
    r = y - X @ beta
    d = X.T @ r / n

    for k in range(1, max_outer + 1):
        T = int(tau * k)
        if T > L:
            break

        beta, active = sdar(X, y, T, beta0=beta, max_iter=max_inner, tol=tol)

        # update dual variable after SDAR
        r = y - X @ beta
        d = X.T @ r / n

        if stop_criterion is not None and stop_criterion(beta, d, k):
            break

    return beta, active