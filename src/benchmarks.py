import numpy as np
# cca-zoo==2.6.0 + scikit-learn==1.4.2 + Python3.11 is a working combination
# other versions MAY have weird dependency issues
# Use for literature review: https://github.com/mikelove/awesome-multi-omics?tab=readme-ov-file
from cca_zoo.linear import SCCA_IPLS, CCA, PLS
import matlab.engine
import os
import sys
sys.path.append(os.path.abspath('/Users/taehyo/Library/CloudStorage/Dropbox/NYU/Research/Research/Code/L0_SCCA_ASDAR'))
from iclr_2022.modelFS import *
from witten_2009._cca_pmd import *
from mai_2019._cca_ipls import *
from nips_2024.local_search import *

# MATLAB engine instance
eng = None

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

    B_hat = model.weights_[0]  
    A_hat = model.weights_[1] 

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
    U, V, D = cca_pmd(X, Y, penaltyx=0.9, penaltyz=0.9, K=K, niter=20, 
        standardize=True)
    B_hat = U
    A_hat = V

    return A_hat, B_hat

def run_ipls(X, Y, K):
    """
    Run iPLS (Mai et al., 2017) for K canonical pairs. (Biometrics 2017)

    Parameters:
        X: (n, p) input matrix (assumed centered)
        Y: (n, q) input matrix (assumed centered)
        K: number of canonical components to extract

    Returns:
        A_hat: (q x K) canonical vectors for Y
        B_hat: (p x K) canonical vectors for X
    """
    X_weights, Y_weights = scca_ipls(X, Y, alpha_lambda_ratio=1.0, beta_lambda_ratio=1.0,
         alpha_lambda=0, beta_lambda=0, niter=100, n_pairs=K, 
         standardize=False, eps=1e-4, glm_impl='pyglmnet')

    A_hat = Y_weights
    B_hat = X_weights
    
    return A_hat, B_hat

def run_l0_cai(X, Y, K):
    """
    Run l0-SCCA (Cai et al., Neurocomputing 2019) via the MATLAB ellzerocca function.

    Parameters:
        X : array, shape (n, p)
            Input data matrix (samples × features), assumed centered.
        Y : array, shape (n, q)
            Input data matrix (samples × features), assumed centered.
        K : int
            Number of canonical components to extract.

    Returns:
        A_hat : ndarray, shape (q, K)
            Canonical weight vectors for Y.
        B_hat : ndarray, shape (p, K)
            Canonical weight vectors for X.

    Installation instructions: https://pypi.org/project/matlabengine/
    """
    def _get_matlab_engine():
        global eng
        if eng is None:
            # Start MATLAB engine
            eng = matlab.engine.start_matlab()
            # Add folder containing ellzerocca.m to MATLAB path
            matlab_path = '/Users/taehyo/Library/CloudStorage/Dropbox/NYU/Research/Research/Code/L0_SCCA_ASDAR/neurocomputing_2019'
            eng.addpath(matlab_path, nargout=0)
        return eng

    # get (and cache) the MATLAB engine
    engine = _get_matlab_engine()

    # transpose & marshal
    Xm = matlab.double(X.T.tolist())
    Ym = matlab.double(Y.T.tolist())

    opts = {'L': K, 'tol_x':1e-5, 'tol_y':1e-5, 'vtol_x':1e-5, 'vtol_y':1e-5, 'deltaL':0}

    SWx_m, SWy_m, _, _ = engine.ellzerocca(Xm, Ym, opts, nargout=4)

    B_hat = np.array(SWx_m)
    A_hat = np.array(SWy_m)
    return A_hat, B_hat

def run_l0_lind(X, Y, K):
    """
    Run l0-SCCA (Lindenbaum et al., 2022) for K canonical pairs. (ICLR 2022)

    Parameters:
        X: (n, p) input matrix (assumed centered)
        Y: (n, q) input matrix (assumed centered)
        K: number of canonical components to extract

    Returns:
        A_hat: (q x K) canonical vectors for Y
        B_hat: (p x K) canonical vectors for X
    """
    def initial_guess(x,y,k):
        n=x.shape[0]
        C_xy=(x.T@y)/(n-1)
        s_xy=C_xy
        thresh=np.percentile(np.abs(C_xy.reshape(-1)),100*(1-k/x.shape[1]))
        C_xy[np.abs(C_xy)<thresh]=0
        [U0,s,V0]=np.linalg.svd(C_xy)
        uu=np.abs(U0[:,0])
        uu[uu<1*thresh]=-0.5
        vv=np.abs(V0[0,:])
        vv[vv<1*thresh]=-0.5
        return uu,vv

    params={'feature_selection': True, 'param_search': False, 'learning_rate': 0.01, 'activation': 'none', 'output_node':K,  'sigma': 0.25, 'display_step': 50, 'hidden_layers_node': [K]}
    uu,vv=initial_guess(X,Y,K)
    params['u']=uu
    params['v']=vv
    params['input_node1'] = X.shape[1]
    params['input_node2'] = Y.shape[1]
    params['batch_size'] = X.shape[0]
    params['lam1'] = 40
    params['lam2'] = 40
    params['learning_rate'] = 0.005
    model = Model(**params)
    model.train(X,Y,X,Y, num_epoch=100)
    tt=model.get_raw_weights()
    B_hat = model.get_prob_alpha()[0][:, np.newaxis] * tt[0][0] 
    A_hat = model.get_prob_alpha()[1][:, np.newaxis] * tt[0][1]  # shape (300, 2)
    return A_hat, B_hat

def run_l0_li(X, Y, K):
    """
    Run l0-SCCA based on localsearch (Li et al., 2023) for K canonical pairs. (NIPS 2024)

    Parameters:
        X: (n, p) input matrix (assumed centered)
        Y: (n, q) input matrix (assumed centered)
        K: number of canonical components to extract

    Returns:
        A_hat: (q x K) canonical vectors for Y
        B_hat: (p x K) canonical vectors for X
    """
    S1, S2, bestf = localsearch(n1, n2, s1, s2, A, B, C)
    
    return A_hat, B_hat

