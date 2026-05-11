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
    DPDataGenerator,
    get_optimal_pft_params,
    pft_algo,
    duchi_algo,
    piecewise_algo,
    hybrid_algo,
    analytic_gaussian_sigma
)
def _parallel_worker(args):
    task_id, dist_name, params_dict, n_users, target_eps, opt_eps0, opt_k, gauss_sigma = args

    np.random.seed((os.getpid() * task_id * 12345) % (2 ** 32 - 1))

    gen = DPDataGenerator(n_samples=n_users)
    if dist_name == 'Normal':
        current_data = gen.get_normal(**params_dict)
    elif dist_name == 'Uniform':
        current_data = gen.get_uniform(**params_dict)
    elif dist_name == 'Exponential':
        current_data = gen.get_exponential(**params_dict)
    elif dist_name == 'Pareto':
        current_data = gen.get_pareto(**params_dict)
    elif dist_name == 'GMM':
        current_data = gen.get_gmm(**params_dict)
    else:
        raise ValueError(f"ERROR: {dist_name}")

    current_true_mean = np.mean(current_data)

    # --- A. DTS  ---
    out_DTS = pft_algo(current_data, opt_eps0, opt_k)
    mse_DTS = (np.mean(out_DTS) - current_true_mean) ** 2

    # --- B. Duchi  ---
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


def run_dynamic_mse_comparison_parallel(dist_name, params_dict, n_users, target_eps, opt_eps0, opt_k, repeats=2000):
    delta = 1e-6
    gauss_sigma = analytic_gaussian_sigma(target_eps, delta, sensitivity=2.0)

    tasks = [
        (i, dist_name, params_dict, n_users, target_eps, opt_eps0, opt_k, gauss_sigma)
        for i in range(1, repeats + 1)
    ]

    num_cores = max(1, mp.cpu_count() - 1)

    total_mses = np.zeros(6)

    with mp.Pool(processes=num_cores) as pool:
        for res in tqdm(pool.imap_unordered(_parallel_worker, tasks),
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
def main_experiment():
    n_users = 1000000
    target_eps_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.5, 3, 3.5, 4]
    repeats = 10000


    generator = DPDataGenerator(n_samples=n_users)


    data_gen_funcs = {
        'Normal': (generator.get_normal, {'mu': 0, 'sigma': 1}),
        'Uniform': (generator.get_uniform, {'low': 0, 'high': 100}),
        'Exponential': (generator.get_exponential, {'lam': 0.5}),
        'Pareto': (generator.get_pareto, {'alpha': 3.0, 'xm': 1.0}),
        'GMM': (generator.get_gmm, {'w1': 0.5, 'mu1': -2, 'mu2': 2, 'sigma1': 1, 'sigma2': 1})
    }

    all_results = []


    for target_eps in target_eps_list:
        print(f"\n=================== [ Target Eps: {target_eps} ] ===================")


        opt_params = get_optimal_pft_params(n_users, target_eps, step=0.0005, verbose=True)
        if opt_params is None:
            print(f"OPT PARAMS NONE")
            continue

        opt_eps0 = opt_params['eps0']
        opt_k = opt_params['k']
        actual_eps_prime = opt_params['eps_prime']


        for dist_name, (gen_func, params_dict) in data_gen_funcs.items():
            tqdm.write(f"  -> RUN: [{dist_name}] ...")


            res_dict = run_dynamic_mse_comparison_parallel(
                dist_name=dist_name,
                params_dict=params_dict,
                n_users=n_users,
                target_eps=target_eps,
                opt_eps0=opt_eps0,
                opt_k=opt_k,
                repeats=repeats
            )


            res_dict['Dataset'] = dist_name
            res_dict['N_Users'] = n_users
            res_dict['Repeats'] = repeats


            res_dict['Dist_Params'] = str(params_dict)


            res_dict['Target_Eps'] = target_eps
            res_dict['Opt_Eps0'] = opt_eps0
            res_dict['Opt_K'] = opt_k
            res_dict['Actual_Eps_Prime'] = actual_eps_prime


            all_results.append(res_dict)

    df_results = pd.DataFrame(all_results)


    info_cols = ['Target_Eps', 'Dataset', 'Dist_Params', 'N_Users', 'Repeats', 'Opt_Eps0', 'Opt_K', 'Actual_Eps_Prime']
    mse_cols = ['DTS', 'Duchi', 'Piecewise', 'Hybrid', 'Laplace', 'Gaussian']
    df_results = df_results[info_cols + mse_cols]


    df_results = df_results.sort_values(by=['Target_Eps', 'Dataset']).reset_index(drop=True)


    print(df_results[['Target_Eps', 'Dataset', 'Opt_Eps0', 'Opt_K', 'DTS', 'Duchi']].head(10))


    save_path = "dp_synthetic_results.csv"
    df_results.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"\n FINISH: {os.path.abspath(save_path)}")

    return df_results


if __name__ == "__main__":
    df = main_experiment()