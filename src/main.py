from simulation import Simulation
from util import evaluate_subspace_error
from model import l0_scca
import benchmarks
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def is_correct_shape(Y, X, A_star, B_star, rho_star):
    print("X shape:", X.shape)         # (n, p)
    print("Y shape:", Y.shape)         # (n, q)
    print("A_star shape:", A_star.shape)  # (p, K)
    print("B_star shape:", B_star.shape)  # (q, K)
    print("True canonical correlations:", rho_star) 

def is_orthogonal(cmat, scov):
    """
    Computes the covariance matrix of canonical variates (e.g. Y*a_1, ..., Y*a_k)
    The off-diganoal elements of A.T @ Σ_YY @ A needs to be 0
    In other words, uncorrelatedness of canonical variates <-> orthonality of canonical vectors under the sample covariance matrix
    """
    print(cmat.T @ scov @ cmat) 

def main():
    # Simulate data
    sim = Simulation(n=200, p=100, seed=42)

    print('Model 1:')
    Y, X, A_star, B_star, rho_star = sim.model1()
    print(A_star[:10])
    print(B_star[:10])
    is_correct_shape(Y, X, A_star, B_star, rho_star)
    is_orthogonal(A_star, sim.Sigma_YY)
    is_orthogonal(B_star, sim.Sigma_XX)
    print('\n')

    print('Model 2:')
    Y, X, A_star, B_star, rho_star = sim.model2()
    print(A_star[:10])
    print(B_star[:10])
    is_correct_shape(Y, X, A_star, B_star, rho_star)
    is_orthogonal(A_star, sim.Sigma_YY)
    is_orthogonal(B_star, sim.Sigma_XX)
    print('\n')

    print('Model 3:')
    Y, X, A_star, B_star, rho_star = sim.model3()
    print(A_star[:10])
    print(B_star[:10])
    is_correct_shape(Y, X, A_star, B_star, rho_star)
    is_orthogonal(A_star, sim.Sigma_YY)
    is_orthogonal(B_star, sim.Sigma_XX)
    print('\n')

    print('Model 4:')
    Y, X, A_star, B_star, rho_star = sim.model4()
    print(A_star[:10])
    print(B_star[:10])
    is_correct_shape(Y, X, A_star, B_star, rho_star)
    is_orthogonal(A_star, sim.Sigma_YY)
    is_orthogonal(B_star, sim.Sigma_XX)
    print('\n')

    print('Model 5:')
    Y, X, A_star, B_star, rho_star = sim.model5()
    print(A_star[:10])
    print(B_star[:10])
    is_correct_shape(Y, X, A_star, B_star, rho_star)
    is_orthogonal(A_star, sim.Sigma_YY)
    is_orthogonal(B_star, sim.Sigma_XX)
    print('\n')

    print('Model 6:')
    Y, X, A_star, B_star, rho_star = sim.model6()
    print(A_star[:10])
    print(B_star[:10])
    is_correct_shape(Y, X, A_star, B_star, rho_star)
    is_orthogonal(A_star, sim.Sigma_YY)
    is_orthogonal(B_star, sim.Sigma_XX)
    print('\n')

    print('Model 7:')
    Y, X, A_star, B_star, rho_star = sim.model7()
    print(A_star[:10])
    print(B_star[:10])
    is_correct_shape(Y, X, A_star, B_star, rho_star)
    is_orthogonal(A_star, sim.Sigma_YY)
    is_orthogonal(B_star, sim.Sigma_XX)
    print('\n')

    print('Model 8:')
    Y, X, A_star, B_star, rho_star = sim.model8()
    print(A_star[:10])
    print(B_star[:10])
    is_correct_shape(Y, X, A_star, B_star, rho_star)
    is_orthogonal(A_star, sim.Sigma_YY)
    is_orthogonal(B_star, sim.Sigma_XX)
    print('\n')
    '''
    # L0 penalties (can be tuned)
    lambda_alpha = [0.1] * A_star.shape[1]
    lambda_beta  = [0.1] * B_star.shape[1]

    
    # Run sparse CCA using SDAR
    A_hat, B_hat = l0_scca(X, Y, K=A_star.shape[1],
                           lambda_alpha_list=lambda_alpha,
                           lambda_beta_list=lambda_beta,
                           mode="sdar")  

    # Evaluate estimated subspaces
    err_A, err_B = evaluate_subspace_error(A_star, B_star, A_hat, B_hat)

    # Print results
    print("True canonical correlations:", rho_star)
    print("Subspace error for A:", err_A)
    print("Subspace error for B:", err_B)
    '''

    # CCA
    print('CCA:') 
    A_hat, B_hat = benchmarks.run_cca(X, Y, A_star.shape[1])
    err_A, err_B = evaluate_subspace_error(A_star, B_star, A_hat, B_hat)
    print("True canonical correlations:", rho_star)
    print("Subspace error for A:", err_A)
    print("Subspace error for B:", err_B)
    print("\n")

    # PMD 
    print('PMD:') 
    A_hat, B_hat = benchmarks.run_pmd(X, Y, A_star.shape[1])
    err_A, err_B = evaluate_subspace_error(A_star, B_star, A_hat, B_hat)
    print("True canonical correlations:", rho_star)
    print("Subspace error for A:", err_A)
    print("Subspace error for B:", err_B)
    print("\n")

    # iPLS 
    print('iPLS:') 
    A_hat, B_hat = benchmarks.run_ipls(X, Y, A_star.shape[1])
    err_A, err_B = evaluate_subspace_error(A_star, B_star, A_hat, B_hat)
    print("True canonical correlations:", rho_star)
    print("Subspace error for A:", err_A)
    print("Subspace error for B:", err_B)
    print("\n")

if __name__ == "__main__":
    main()