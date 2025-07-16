from simulation import Simulation
import util
import model
import benchmarks
import numpy as np
import pandas as pd
import os
from datetime import datetime
import warnings
from collections import defaultdict
import time
from tqdm import tqdm
import argparse 

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

def finetune():
    # Configurations
    n_p_combinations = [(500, 300), (500, 600), (500, 1000), (500, 1200), (500, 1500), (500, 2000)]
    penaltyxs = np.linspace(0.1, 0.5, 5)
    penaltyzs = np.linspace(0.1, 0.5, 5)
    alpha_lambdas = np.linspace(0.1, 0.9, 5)
    beta_lambdas = np.linspace(0.1, 0.9, 5)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'finetune_results_{timestamp}.txt'

    with open(output_file, 'w') as f:
        for n, p in tqdm(n_p_combinations, desc="finetuning progress"):            
            for i in range(1, 9):
                sim_val = Simulation(n=n, p=p, seed=999)
                model_func_val = getattr(sim_val, f'model{i}')

                Y_val, X_val, A_star, B_star, rho_star = model_func_val()
                K = A_star.shape[1]

                # Finetune PMD
                best_score = -np.inf
                best_params = (None, None)
                for px in penaltyxs:
                    for pz in penaltyzs:
                        A_hat, B_hat = benchmarks.run_pmd(X_val, Y_val, K, penaltyx=px, penaltyz=pz)
                        score = util.evaluate_canonical_correlation(A_hat, B_hat, 
                                                                Y_val.T @ Y_val / n, 
                                                                X_val.T @ X_val / n,
                                                                Y_val.T @ X_val / n)
                        if np.sum(np.square(score)) > best_score:
                            best_score = np.sum(np.square(score))
                            best_params = (px, pz)
                            print(best_score, best_params)
                f.write(f"[PMD] n={n}, p={p}, model={i}, best_penaltyx={best_params[0]:.2f}, best_penaltyz={best_params[1]:.2f}, cc={best_score}\n")
                f.flush()

                # Finetune iPLS
                best_score = -np.inf
                best_params = (None, None)
                for ax in alpha_lambdas:
                    for az in beta_lambdas:
                        A_hat, B_hat = benchmarks.run_ipls(X_val, Y_val, K, alpha_lambda=ax, beta_lambda=az)
                        score = util.evaluate_canonical_correlation(A_hat, B_hat, 
                                                                Y_val.T @ Y_val / n, 
                                                                X_val.T @ X_val / n,
                                                                Y_val.T @ X_val / n)
                        if np.sum(np.square(score)) > best_score:
                            best_score = np.sum(np.square(score))
                            best_params = (ax, az)
                            print(best_score, best_params)
                f.write(f"[iPLS] n={n}, p={p}, model={i}, best_alpha_lambda={best_params[0]:.2f}, best_beta_lambda={best_params[1]:.2f}, cc={best_score}\n")
                f.flush()

# time taken should be with or without finetuning step? some methods automatically find the best parameters
# start 200 replications with the found parameters 
# study and write manuscript in the meantime
def sim():
    ######### CONFIG
    N_rep = 1
    n_p_combinations = [(500, 2000)] 
    # complete set:  
    # (500,300) sim_results_20250628_125023.csv
    # (500, 600) sim_results_20250628_150000.csv
    # (500, 1000) sim_results_20250628_150000.csv
    # (500, 1200) sim_results_20250628_150000.csv
    # (500, 1500) sim_results_20250628_150000.csv
    # (500, 2000) Model 1-3 sim_results_20250628_150000.csv
    # (500, 2000) Model 4-8 sim_results_20250629_205052.csv
    #########

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'sim_results_{timestamp}.csv'

    # the order has to match the functions 
    metrics = [
        'rho_hat',
        'subspace_error_A', 'subspace_error_B',
        'cosine_similarity_A', 'cosine_similarity_B',
        'sensitivity_A', 'specificity_A', 'precision_A', 'f1_A', 'mcc_A',
        'sensitivity_B', 'specificity_B', 'precision_B', 'f1_B', 'mcc_B',
        'time'
    ]

    columns = ['n', 'p', 'model', 'method', 'rho_star'] + metrics
    if not os.path.exists(output_file):
        pd.DataFrame(columns=columns).to_csv(output_file, index=False)

    for n, p in n_p_combinations:
        sim = Simulation(n=n, p=p, seed=42)

        for i in range(8, 9):
            print(f"\n=== Running Simulation Model {i} with n={n}, p={p} ===")
            model_func = getattr(sim, f'model{i}')

            for method_name, method_func in tqdm({
                    'CCA': benchmarks.run_cca,
                    'PMD': benchmarks.run_pmd,
                    'L0_li': benchmarks.run_l0_li,
                    'L0_cai': benchmarks.run_l0_cai,
                    'L0_lind': benchmarks.run_l0_lind,
                    'iPLS': benchmarks.run_ipls,
                    'iPLS_L0': model.run_l0_ipls}.items(), desc=f"Methods", leave=False):
                metrics = defaultdict(list)

                for rep in range(N_rep):
                    Y, X, A_star, B_star, rho_star = model_func()

                    try:
                        start = time.time()
                        A_hat, B_hat = method_func(X, Y, A_star.shape[1]) # choosing of K would be up to users for RDA 
                        elapsed = time.time() - start

                        err_A, err_B = util.evaluate_subspace_error(A_star, B_star, A_hat, B_hat)
                        cos_A, cos_B = util.evaluate_cossim(A_star, B_star, A_hat, B_hat)
                        cc = util.evaluate_canonical_correlation(A_hat, B_hat, sim.Sigma_YY, sim.Sigma_XX, sim.Sigma_YX)
                    except Exception:
                        err_A = err_B = np.sqrt(2 * A_star.shape[1]) # maximum subspace error
                        cos_A = cos_B = 0.0 # minimum abs cossim
                        cc = [0.0] * A_star.shape[1] # minimum canonical correlation
                    support_metrics = util.evaluate_support_metrics(A_star, B_star, A_hat, B_hat)

                    metrics['rho_hat'].append(cc)
                    metrics['subspace_error_A'].append(err_A)
                    metrics['subspace_error_B'].append(err_B)
                    metrics['cosine_similarity_A'].append(cos_A)
                    metrics['cosine_similarity_B'].append(cos_B)
                    for k, v in support_metrics.items():
                        metrics[k].append(v)
                    metrics['time'].append(elapsed)

                summary = {
                    'n': n,
                    'p': p,
                    'model': f'Model {i}',
                    'method': method_name,
                    'rho_star': str(list(rho_star))
                }

                for k, v in metrics.items():
                    if k == 'rho_hat':
                        cc_array = np.array(v)  # shape: (N, K)
                        rho_hat_strs = [
                            f"{np.mean(cc_array[:, j]):.4f} ({np.std(cc_array[:, j]):.4f})"
                            for j in range(cc_array.shape[1])
                        ]
                        summary[k] = "[" + ", ".join(rho_hat_strs) + "]"
                    else:
                        mean = np.mean(v)
                        std = np.std(v)
                        summary[k] = f"{mean:.4f} ({std:.4f})"

                pd.DataFrame([summary]).to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FcHMRF-LIS for spatial multiple hypothesis testing.')
    parser.add_argument('--mode', default='sim', type=str, help='sim/real/finetune')
    args = parser.parse_args()

    if args.mode == 'sim':
        sim()
    elif args.mode == 'real':
        pass
    elif args.mode == 'finetune':
        finetune()