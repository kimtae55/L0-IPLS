import numpy as np
from joblib import Parallel, delayed

def sdar(X, y, T=5, alpha=0.0, max_iter=100):
    """
    SDAR implementation based on provided MATLAB code

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
        Design matrix.
    y : np.ndarray, shape (n,)
        Response vector.
    T : int
        Sparsity level (number of features to select).
    alpha : float, optional
        Regularization parameter (default 0.0).
    max_iter : int, optional
        Maximum number of iterations.

    Returns
    -------
    beta_k : np.ndarray, shape (p,)
        Final coefficient estimate β̂.
    """
    n, p = X.shape

    # Precompute Gram matrix and X^T y
    Gram = X.T @ X                      # p × p
    Xty = X.T @ y                       # p x p

    # Require: β^0 = 0, d^0 = X^T(y - Xβ^0)/n = X^T y / n
    beta_k = np.zeros(p)                # β^k
    d_k = Xty / n                       # d^k

    S = np.arange(p)                    # Full index set

    for k in range(max_iter):
        # 2: Compute p^k = β^k + d^k
        p_k = beta_k + d_k

        # Find the T-th largest |p_k| via argpartition (O(p))
        tau = np.partition(np.abs(p_k), -T)[-T]

        # Active set A^k = { i : |p_k[i]| >= tau }
        mask_A_k = np.abs(p_k) >= tau
        A_k = np.nonzero(mask_A_k)[0]

        # Inactive set I^k = S \ A^k
        mask_I_k = ~mask_A_k
        I_k = np.nonzero(mask_I_k)[0]

        # 3-4: Initialize β^{k+1} and d^{k+1}
        beta_next = np.zeros(p)
        d_next = np.zeros(p)

        # 5: β^{k+1}_{A^k} = (X_{A^k}^T X_{A^k})^{-1} X_{A^k}^T y
        G = Gram[np.ix_(A_k, A_k)] + alpha * np.eye(len(A_k))
        beta_next[A_k] = np.linalg.solve(G, Xty[A_k])

        # 6: d^{k+1}_{I^k} = X_{I^k}^T (y - X_{A^k} β^{k+1}_{A^k}) / n
        residual = y - X[:, A_k] @ beta_next[A_k]
        d_next[I_k] = (X[:, I_k].T @ residual) / n

        # Check convergence: compute A^{k+1} from p^{k+1}
        p_next = beta_next + d_next
        tau_next = np.partition(np.abs(p_next), -T)[-T]
        A_next = np.nonzero(np.abs(p_next) >= tau_next)[0]

        # Update current betas
        beta_k, d_k = beta_next, d_next

        # If A^{k+1} == A^k, stop
        if np.array_equal(np.sort(A_next), np.sort(A_k)):
            beta_k, d_k = beta_next, d_next
            break

    return beta_k

def compute_hbic(T, X, y, n, p, alpha, max_iter):
    beta_T = sdar(X, y, T=T, alpha=alpha, max_iter=max_iter)
    residual = y - X @ beta_T
    rss = np.maximum(np.linalg.norm(residual)**2 / n, 1e-8)
    penalty = T * np.log(p - 1) * np.log(np.log(n))
    hbic_T = n * np.log(rss) + penalty
    return hbic_T, beta_T

def asdar(X, y, max_iter_per_sdar, L=None, alpha=1e-5, mode='parallel', n_jobs=6):
    """
    ASDAR using HBIC selection with optional parallel or serial execution.

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
        Design matrix.
    y : np.ndarray, shape (n,)
        Response vector.
    L : int, optional
        Maximum support size T to consider. If None, set to floor(n / log(n)).
    max_iter_per_sdar : int, optional
        Max iterations per SDAR run. If None, use T for each run.
    mode : str, optional
        'parallel' (default) or 'single' to control execution mode.
    n_jobs : int, optional
        Number of parallel jobs to run (only used if mode='parallel').

    Returns
    -------
    beta_best : np.ndarray
        Estimated coefficients corresponding to the best HBIC.
    """
    n, p = X.shape
    if L is None:
        L = int(np.floor(n / np.log(n)))

    tasks = range(1, L + 1) # could change this to be more sparse to speed up, using 'mode=fast'

    if mode == 'parallel':
        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_hbic)(T, X, y, n, p, alpha, max_iter_per_sdar)
            for T in tasks
        )
    elif mode == 'single':
        results = []
        for T in tasks:
            results.append(compute_hbic(T, X, y, n, p, alpha, max_iter_per_sdar))
    else:
        raise ValueError("mode must be 'parallel' or 'single'")

    best_index = int(np.argmin([hbic for hbic, _ in results]))
    return results[best_index][1]