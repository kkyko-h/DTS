import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm



from core import (
    get_optimal_pft_params,
    pft_algo,
    duchi_algo,
    piecewise_algo,
    hybrid_algo,
    analytic_gaussian_sigma,
    DataHandler
)
def _parallel_worker_real(args):
    task_id, current_data, n_users, target_eps, opt_eps0, opt_k, gauss_sigma = args


    np.random.seed((os.getpid() * task_id * 12345) % (2 ** 32 - 1))

    current_true_mean = np.mean(current_data)

    # --- A. DTS  ---
    out_DTS = pft_algo(current_data, opt_eps0, opt_k)
    mse_DTS = (np.mean(out_DTS) - current_true_mean) ** 2

    # --- B. Duchi ---
    out_duchi = duchi_algo(current_data, target_eps)
    mse_duchi = (np.mean(out_duchi) - current_true_mean) ** 2

    # --- C. Piecewise  ---
    out_piece = piecewise_algo(current_data, target_eps)
    mse_piece = (np.mean(out_piece) - current_true_mean) ** 2

    # --- D. Hybrid  ---
    out_hybrid = hybrid_algo(current_data, target_eps)
    mse_hybrid = (np.mean(out_hybrid) - current_true_mean) ** 2

    # --- E. Laplace  ---
    lap_noise = np.random.laplace(loc=0.0, scale=2.0 / target_eps, size=n_users)
    mse_laplace = (np.mean(current_data + lap_noise) - current_true_mean) ** 2

    # --- F. Gaussian  ---
    gauss_noise = np.random.normal(loc=0.0, scale=gauss_sigma, size=n_users)
    mse_gaussian = (np.mean(current_data + gauss_noise) - current_true_mean) ** 2

    return np.array([mse_DTS, mse_duchi, mse_piece, mse_hybrid, mse_laplace, mse_gaussian])



def run_real_mse_comparison_parallel(current_data, target_eps, opt_eps0, opt_k, repeats=2000):
    n_users = len(current_data)
    delta = 1e-6
    gauss_sigma = analytic_gaussian_sigma(target_eps, delta, sensitivity=2.0)


    tasks = [
        (i, current_data, n_users, target_eps, opt_eps0, opt_k, gauss_sigma)
        for i in range(1, repeats + 1)
    ]

    num_cores = max(1, mp.cpu_count() - 1)
    total_mses = np.zeros(6)

    with mp.Pool(processes=num_cores) as pool:
        for res in tqdm(pool.imap_unordered(_parallel_worker_real, tasks),
                        total=repeats, desc=f"Eval eps={target_eps}", leave=False, ncols=100):
            total_mses += res

    avg_mses = total_mses / repeats

    return {
        'DTS': avg_mses[0],
        'Duchi': avg_mses[1],
        'Piecewise': avg_mses[2],
        'Hybrid': avg_mses[3],
        'Laplace': avg_mses[4],
        'Gaussian': avg_mses[5]
    }

def main_experiment_real():
    # 1. ACS Income
    # FILE_PATH = "./data/psam_p06.csv"
    # TARGET_COL = "PINCP"

    # 2. NYC Green Taxi
    # FILE_PATH = "./data/green_tripdata_2018-01.parquet"
    # TARGET_COL = "trip_distance"

    # 3. SF Employee Compensation
    FILE_PATH = "./data/Employee_Compensation.csv"
    TARGET_COL = "Total Compensation"


    target_eps_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2,2.5,3,3.5,4]
    repeats = 10000


    handler = DataHandler(FILE_PATH)
    handler.load_data(columns=[TARGET_COL], sample_frac=1.0)

    real_data_norm = handler.preprocess_column(TARGET_COL, method='min-max', clip_quantile=(0, 1))

    if real_data_norm is None:
        print("DATA NONE")
        return

    n_users = len(real_data_norm)

    all_results = []


    for target_eps in target_eps_list:
        print(f"\n=================== [ Target Eps: {target_eps} ] ===================")

        opt_params = get_optimal_pft_params(n_users, target_eps, step=0.0005, verbose=False)
        if opt_params is None:
            print(f"OPT PARAM NONE")
            continue

        opt_eps0 = opt_params['eps0']
        opt_k = opt_params['k']
        actual_eps_prime = opt_params['eps_prime']


        res_dict = run_real_mse_comparison_parallel(
            current_data=real_data_norm,
            target_eps=target_eps,
            opt_eps0=opt_eps0,
            opt_k=opt_k,
            repeats=repeats
        )


        res_dict['Dataset'] = f"{TARGET_COL}"
        res_dict['N_Users'] = n_users
        res_dict['Repeats'] = repeats
        res_dict['Target_Eps'] = target_eps
        res_dict['Opt_Eps0'] = opt_eps0
        res_dict['Opt_K'] = opt_k
        res_dict['Actual_Eps_Prime'] = actual_eps_prime


        meta = handler.meta_info[TARGET_COL]
        res_dict['True_Mean_RealWorld'] = meta['true_mean_real']
        res_dict['Scale'] = meta['scale']
        res_dict['Shift'] = meta['shift']

        all_results.append(res_dict)


    df_results = pd.DataFrame(all_results)

    info_cols = ['Target_Eps', 'Dataset', 'N_Users', 'Repeats', 'Opt_Eps0', 'Opt_K', 'Actual_Eps_Prime']
    mse_cols = ['DTS', 'Duchi', 'Piecewise', 'Hybrid', 'Laplace', 'Gaussian']
    df_results = df_results[info_cols + mse_cols]

    print(df_results[['Target_Eps', 'Dataset', 'Opt_Eps0', 'Opt_K', 'DTS', 'Duchi']].head())

    save_path = "dp_real_world_results_Employee_Compensation.csv"
    df_results.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"\n FINISH: {os.path.abspath(save_path)}")

    return df_results


if __name__ == "__main__":
    main_experiment_real()