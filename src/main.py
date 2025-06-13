from simulation import Simulation
import util
import model
import benchmarks
import numpy as np
import pandas as pd
import os
from datetime import datetime
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

# do 200 replications 
def main():
    n_p_combinations = [(500, 300), (500, 1000), (500, 1200), (500, 2000)]

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'sim_results_{timestamp}.csv'

    if not os.path.exists(output_file):
        pd.DataFrame(columns=['n', 'p', 'model', 'method',
                                'subspace_error_A', 'subspace_error_B',
                                'mcc_A', 'mcc_B',
                                'cosine_similarity_A', 'cosine_similarity_B',
                                'rho_star'
                                ]).to_csv(output_file, index=False)

    for n, p in n_p_combinations:
        sim = Simulation(n=n, p=p, seed=42)
        for i in range(1, 9):
            print(f"\n=== Running Simulation Model {i} with n={n}, p={p} ===")
            model_func = getattr(sim, f'model{i}')
            Y, X, A_star, B_star, rho_star = model_func()

            is_correct_shape(Y, X, A_star, B_star, rho_star)
            is_orthogonal(A_star, sim.Sigma_YY)
            is_orthogonal(B_star, sim.Sigma_XX)

            for method_name, method_func in {
                'CCA': benchmarks.run_cca,
                'PMD': benchmarks.run_pmd,
                'L0_cai': benchmarks.run_l0_cai,
                'L0_lind': benchmarks.run_l0_lind,
                'iPLS': benchmarks.run_ipls,
                'iPLS_L0': model.run_l0_ipls,
            }.items():
                A_hat, B_hat = method_func(X, Y, A_star.shape[1])
                err_A, err_B = util.evaluate_subspace_error(A_star, B_star, A_hat, B_hat)
                mcc_A, mcc_B = util.evaluate_mcc(A_star, B_star, A_hat, B_hat)
                cos_A, cos_B = util.evaluate_cossim(A_star, B_star, A_hat, B_hat)

                # Save all metrics to CSV
                row = pd.DataFrame([{
                    'n': n,
                    'p': p,
                    'model': f'Model {i}',
                    'method': method_name,
                    'subspace_error_A': err_A,
                    'subspace_error_B': err_B,
                    'mcc_A': mcc_A,
                    'mcc_B': mcc_B,
                    'cosine_similarity_A': cos_A,
                    'cosine_similarity_B': cos_B,
                    'rho_star': str(list(rho_star))  # safe for CSV
                }])

                row.to_csv(output_file, mode='a', header=False, index=False)

if __name__ == "__main__":
    main()