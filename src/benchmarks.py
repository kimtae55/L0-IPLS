import numpy as np
# cca-zoo==2.6.0 + scikit-learn==1.4.2 + Python3.11 is a working combination
# other versions MAY have weird dependency issues
from cca_zoo.linear import SCCA_IPLS, CCA, PLS

def run_cca(X, Y, K):
    """
    Run classical CCA for K canonical pairs.

    Parameters:
        X: (n, p) input matrix (assumed centered)
        Y: (n, q) input matrix (assumed centered)
        K: number of canonical components to extract

    Returns:
        A_hat: (q x K) canonical vectors for Y
        B_hat: (p x K) canonical vectors for X
    """
    model = CCA(latent_dimensions=K)
    model.fit([X, Y])

    A_hat = model.weights_[0]  # shape: (q, K)
    B_hat = model.weights_[1]  # shape: (p, K)

    return A_hat, B_hat

def run_pmd(X, Y, K):
    """
    Run PMD (Witten et al., 2009) for K canonical pairs using cca-zoo.

    Parameters:
        X: (n, p) input matrix (assumed centered)
        Y: (n, q) input matrix (assumed centered)
        K: number of canonical components to extract

    Returns:
        A_hat: (q x K) canonical vectors for Y
        B_hat: (p x K) canonical vectors for X
    """
    model = PLS(latent_dimensions=K) # sparse PLS <-> PMD
    model.fit([X, Y])

    A_hat = model.weights_[1]  # Canonical vectors for Y
    B_hat = model.weights_[0]  # Canonical vectors for X

    return A_hat, B_hat

def run_colar(X, Y, K):
    """
    Run CoLaR (Gao et al., 2017) for K canonical pairs.

    Parameters:
        X: (n, p) input matrix (assumed centered)
        Y: (n, q) input matrix (assumed centered)
        K: number of canonical components to extract

    Returns:
        A_hat: (q x K) canonical vectors for Y
        B_hat: (p x K) canonical vectors for X
    """
    pass

def run_ipls(X, Y, K):
    """
    Run iPLS (Mai et al., 2017) for K canonical pairs.

    Parameters:
        X: (n, p) input matrix (assumed centered)
        Y: (n, q) input matrix (assumed centered)
        K: number of canonical components to extract

    Returns:
        A_hat: (q x K) canonical vectors for Y
        B_hat: (p x K) canonical vectors for X
    """
    model = SCCA_IPLS(latent_dimensions=K)
    model.fit([X, Y])

    A_hat = model.weights_[0]  # shape: (q, K)
    B_hat = model.weights_[1]  # shape: (p, K)

    return A_hat, B_hat

def run_l0_cai(X, Y, K):
    """
    Run l0-SCCA (Cai et al., 2019) for K canonical pairs.

    Parameters:
        X: (n, p) input matrix (assumed centered)
        Y: (n, q) input matrix (assumed centered)
        K: number of canonical components to extract

    Returns:
        A_hat: (q x K) canonical vectors for Y
        B_hat: (p x K) canonical vectors for X
    """
    pass

def run_l0_lind(X, Y, K):
    """
    Run l0-SCCA (Lindenbaum et al., 2021) for K canonical pairs.

    Parameters:
        X: (n, p) input matrix (assumed centered)
        Y: (n, q) input matrix (assumed centered)
        K: number of canonical components to extract

    Returns:
        A_hat: (q x K) canonical vectors for Y
        B_hat: (p x K) canonical vectors for X
    """
    pass

def run_l0_li(X, Y, K):
    """
    Run l0-SCCA (Li et al., 2023) for K canonical pairs.

    Parameters:
        X: (n, p) input matrix (assumed centered)
        Y: (n, q) input matrix (assumed centered)
        K: number of canonical components to extract

    Returns:
        A_hat: (q x K) canonical vectors for Y
        B_hat: (p x K) canonical vectors for X
    """
    pass