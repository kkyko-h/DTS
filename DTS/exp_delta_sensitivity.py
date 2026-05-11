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
    DataHandler
)

def _parallel_worker_DTS_only(args):

    task_id, current_data, opt_eps0, opt_k = args

    np.random.seed((os.getpid() * task_id * 12345) % (2 ** 32 - 1))

    current_true_mean = np.mean(current_data)


    out_DTS = pft_algo(current_data, opt_eps0, opt_k)
    mse_DTS = (np.mean(out_DTS) - current_true_mean) ** 2

    return mse_DTS

def run_DTS_only_parallel(current_data, opt_eps0, opt_k, repeats):

    tasks = [
        (i, current_data, opt_eps0, opt_k)
        for i in range(1, repeats + 1)
    ]

    num_cores = max(1, mp.cpu_count() - 1)
    total_mse_DTS = 0.0

    with mp.Pool(processes=num_cores) as pool:
        for res in tqdm(pool.imap_unordered(_parallel_worker_DTS_only, tasks),
                        total=repeats, desc=f"Evaluating", leave=False, ncols=100):
            total_mse_DTS += res

    return total_mse_DTS / repeats


def main_experiment_delta_impact():
    # FILE_PATH = r"D:\project\finale\data\income\psam_p06.csv"
    # TARGET_COL = "PINCP"
    # FILE_PATH = r"D:\project\finale\texi\green_tripdata_2018-01.parquet"
    # TARGET_COL = 'trip_distance'
    FILE_PATH = r"D:\project\finale\data\Employee\Employee_Compensation_20260126.csv"
    TARGET_COL = "Total Compensation"


    target_eps_list = [0.1,0.5,1,1.5,2,2.5,3,3.5,4]  # 你可以自己增减

    delta_list = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]

    repeats = 10000


    handler = DataHandler(FILE_PATH)
    handler.load_data(columns=[TARGET_COL], sample_frac=1.0)
    real_data_norm = handler.preprocess_column(TARGET_COL, method='min-max', clip_quantile=(0, 1))

    if real_data_norm is None:
        return

    n_users = len(real_data_norm)

    all_results = []


    total_tasks = len(target_eps_list) * len(delta_list)
    global_pbar = tqdm(total=total_tasks, desc="Delta RUN", position=0)

    for target_eps in target_eps_list:
        for current_delta in delta_list:
            tqdm.write(f"\n=== [ Target Eps: {target_eps} | Delta: {current_delta:.1e} ] ===")


            opt_params = get_optimal_pft_params(
                n_users,
                target_eps,
                delta_target=current_delta,
                step=0.0005,
                verbose=False
            )

            if opt_params is None:
                tqdm.write(f"OPT PARAMS NOT FOUND!\n")
                global_pbar.update(1)
                continue

            opt_eps0 = opt_params['eps0']
            opt_k = opt_params['k']


            mse_DTS = run_DTS_only_parallel(
                current_data=real_data_norm,
                opt_eps0=opt_eps0,
                opt_k=opt_k,
                repeats=repeats
            )


            all_results.append({
                'Dataset': TARGET_COL,
                'N_Users': n_users,
                'Repeats': repeats,
                'Target_Eps': target_eps,
                'Target_Delta': current_delta,
                'Opt_Eps0': opt_eps0,
                'Opt_K': opt_k,
                'Actual_Eps_Prime': opt_params['eps_prime'],
                'MSE_DTS': mse_DTS
            })

            global_pbar.update(1)

    global_pbar.close()


    df_results = pd.DataFrame(all_results)

    print("\n✅ FINISH：")
    print(df_results[['Target_Eps', 'Target_Delta', 'Opt_Eps0', 'Opt_K', 'MSE_DTS']].head(10))

    save_path = "dp_DTS_delta_impact_Employee_Compensation.csv"
    df_results.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 FINISH: {os.path.abspath(save_path)}")

    return df_results


if __name__ == "__main__":
    main_experiment_delta_impact()