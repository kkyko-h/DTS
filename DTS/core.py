import numpy as np
import math
from scipy.special import gammaln
from scipy.stats import norm
from scipy.optimize import brentq
import pandas as pd
class DPDataGenerator:
    def __init__(self, n_samples=10000):
        self.n = n_samples

    def _normalize(self, data):

        d_min = np.min(data)
        d_max = np.max(data)
        norm_data = 2 * (data - d_min) / (d_max - d_min) - 1
        return norm_data

    def get_normal(self, mu=0, sigma=1):

        data = np.random.normal(mu, sigma, self.n)
        return self._normalize(data)

    def get_uniform(self, low=0, high=100):

        data = np.random.uniform(low, high, self.n)
        return self._normalize(data)

    def get_exponential(self, lam=1.0):


        data = np.random.exponential(1/lam, self.n)
        # print("exponential max =",np.max(data) )
        return self._normalize(data)

    def get_pareto(self, alpha=2.0, xm=1.0):


        data = (np.random.pareto(alpha, self.n) + 1) * xm
        # print("pareto max =",np.max(data) )
        return self._normalize(data)

    def get_gmm(self, w1=0.5, mu1=-5, mu2=5, sigma1=1, sigma2=1):

        n1 = int(self.n * w1)
        n2 = self.n - n1
        data1 = np.random.normal(mu1, sigma1, n1)
        data2 = np.random.normal(mu2, sigma2, n2)
        data = np.concatenate([data1, data2])
        np.random.shuffle(data) # 打乱顺序
        return self._normalize(data)






def _log_comb_vec(n, k_arr):
    n = float(n)
    mask = (k_arr >= 0) & (k_arr <= n)
    k_safe = np.where(mask, k_arr, 0.0)
    res = gammaln(n + 1) - gammaln(k_safe + 1) - gammaln(n - k_safe + 1)
    res[~mask] = -np.inf
    return res


def _solve_pft_vectorized(n, k, epsilon0, delta_target, alpha=0.001):
    N = int(n + k)
    p = math.exp(epsilon0) / (math.exp(epsilon0) + 1)
    Np = int(round(N * p))
    M = int(k + 1)

    log_P_T = math.log(alpha * delta_target) - 2 * math.log(M + 1)
    if log_P_T < -700: log_P_T = -700

    log_denom_hyper = _log_comb_vec(N, np.array([M]))[0]

    x_range = np.arange(0, M + 1)
    lp_arr = _log_comb_vec(Np, x_range) + _log_comb_vec(N - Np, M - x_range) - log_denom_hyper

    mask = lp_arr >= log_P_T
    a1_arr = x_range[mask]
    lp1_arr = lp_arr[mask]

    a2_arr = M - a1_arr
    lp2_arr = lp1_arr
    lp_joint_mat = lp1_arr[:, None] + lp2_arr[None, :]

    with np.errstate(divide='ignore', invalid='ignore'):
        log_a1 = np.log(a1_arr.astype(np.float64))
        log_a2 = np.log(a2_arr.astype(np.float64))
        log_b1 = np.log((M - a1_arr).astype(np.float64))
        log_b2 = np.log((M - a2_arr).astype(np.float64))

        term1 = np.abs(log_a1[:, None] - log_a2[None, :])
        term2 = np.abs(log_b1[:, None] - log_b2[None, :])
        np.nan_to_num(term1, copy=False, nan=0.0)
        np.nan_to_num(term2, copy=False, nan=0.0)
        loss_mat = np.maximum(term1, term2)

    loss_flat = loss_mat.ravel()
    prob_flat = lp_joint_mat.ravel()

    sort_idx = np.argsort(loss_flat)
    loss_sorted = loss_flat[sort_idx]
    prob_sorted = prob_flat[sort_idx]

    prob_exp = np.exp(prob_sorted)
    total_kept_mass = np.sum(prob_exp)
    target_coverage = 1.0 - delta_target


    tolerance = delta_target * 1e-3
    if total_kept_mass < target_coverage - tolerance:
        return None

    cum_prob = np.cumsum(prob_exp)


    search_target = min(target_coverage, total_kept_mass - (delta_target * 1e-5))

    idx = np.searchsorted(cum_prob, search_target)
    if idx >= len(loss_sorted):
        idx = len(loss_sorted) - 1
    return loss_sorted[idx]


def _find_min_k_for_eps0(n, eps0, target_eps, delta_target, max_k_limit=500000):
    k_high = 10
    while True:
        if k_high > max_k_limit:
            return None, None
        eps_prime = _solve_pft_vectorized(n, k_high, eps0, delta_target)
        if eps_prime is not None and eps_prime <= target_eps:
            break
        k_high *= 2

    k_low = k_high // 2
    best_k = k_high
    best_eps_prime = eps_prime

    while k_low <= k_high:
        k_mid = (k_low + k_high) // 2
        current_eps_prime = _solve_pft_vectorized(n, k_mid, eps0, delta_target)

        if current_eps_prime is not None and current_eps_prime <= target_eps:
            best_k = k_mid
            best_eps_prime = current_eps_prime
            k_high = k_mid - 1
        else:
            k_low = k_mid + 1

    return best_k, best_eps_prime


def get_optimal_pft_params(n, target_eps, delta_target=1e-6, step=0.05, verbose=False):

    best_score = float('inf')
    best_params = None

    eps0_candidates = np.arange(target_eps - step, 0.05, -step)

    for eps0 in eps0_candidates:
        best_k, actual_eps_prime = _find_min_k_for_eps0(n, eps0, target_eps, delta_target)

        if best_k is None:
            continue

        N = n + best_k
        ex = math.exp(eps0)
        A = (ex + 1) / (ex - 1)
        variance = A ** 2 - 1

        fpc = (N - 0.5 * n) / (N - 1)
        score = fpc * variance

        if score < best_score:
            best_score = score
            best_params = {
                'eps0': eps0,
                'k': best_k,
                'eps_prime': actual_eps_prime,
                'fpc': fpc,
                'variance': variance,
                'score': score
            }



    return best_params


def pft_algo(data, epsilon0, k):

    n = len(data)


    p_fpc = np.exp(epsilon0) / (np.exp(epsilon0) + 1)
    A_fpc = (np.exp(epsilon0) + 1) / (np.exp(epsilon0) - 1)

    N = int(n + k)
    Np = int(round(N * p_fpc))


    pm = np.concatenate([np.ones(Np, dtype=np.int8), -np.ones(N - Np, dtype=np.int8)])
    pn = np.concatenate([np.ones(N - Np, dtype=np.int8), -np.ones(Np, dtype=np.int8)])


    np.random.shuffle(pm)
    np.random.shuffle(pn)


    probs = (1 + data) / 2
    rand_round = np.random.random(n)
    binary_data = np.where(rand_round < probs, 1, -1)


    out_fpc = np.where(binary_data == 1, pm[:n], pn[:n]) * A_fpc
    return out_fpc


def duchi_algo(data, epsilon):

    exp_eps = np.exp(epsilon)
    C = (exp_eps + 1) / (exp_eps - 1)


    probs = 0.5 * (1 + data / C)
    bits = np.random.random(data.shape) < probs


    return C * (2 * bits - 1)


def piecewise_algo(data, epsilon):

    C = (np.exp(epsilon / 2) + 1) / (np.exp(epsilon / 2) - 1)
    l_data = (C + 1) / 2 * data - (C - 1) / 2
    r_data = l_data + C - 1


    p_1 = np.exp(epsilon / 2) / (np.exp(epsilon / 2) + 1) * np.ones(data.size)
    p_1 = np.clip(p_1, 0.0, 1.0)
    ber_1 = np.random.binomial(1, p_1, size=None)
    t_1 = np.random.uniform(low=l_data, high=r_data, size=data.size)


    p_2 = (l_data + C) / (C + 1)
    p_2 = np.clip(p_2, 0.0, 1.0)
    ber_2 = np.random.binomial(1, p_2, size=None)
    t_2 = np.random.uniform(low=-C, high=l_data, size=data.size)
    t_3 = np.random.uniform(low=r_data, high=C, size=data.size)


    t = np.multiply(ber_1, t_1) + np.multiply(1 - ber_1, np.multiply(ber_2, t_2) + np.multiply(1 - ber_2, t_3))
    return t


def get_epsilon_star():
    return -np.log(27) + np.log(-5 + 2 * np.power(6353 - 405 * np.power(241, 0.5), 1. / 3) + \
                                2 * np.power(6353 + 405 * np.power(241, 0.5), 1. / 3))


def hybrid_algo(data, epsilon):
    epsilon_star = get_epsilon_star()
    alpha = 0


    if epsilon > epsilon_star:
        alpha = 1 - np.exp(-epsilon / 2)

    p_piecewise = np.ones(data.size) * alpha
    ber_piecewise = np.random.binomial(1, p_piecewise, size=None)

    return np.multiply(ber_piecewise, piecewise_algo(data, epsilon)) + \
        np.multiply(1 - ber_piecewise, duchi_algo(data, epsilon))


def laplace_mech_ldp(data, epsilon):
    noise = np.random.laplace(loc=0.0, scale=2.0 / epsilon, size=data.shape)
    return data + noise


def analytic_gaussian_sigma(epsilon, delta, sensitivity):

    def _delta(sigma):
        if sigma <= 0: return 1.0
        s = sensitivity
        a = (s / sigma) / 2
        b = epsilon / (s / sigma)
        val = norm.cdf(a - b) - np.exp(epsilon) * norm.cdf(-a - b)
        return val

    try:
        sigma = brentq(lambda s: _delta(s) - delta, 1e-5, 100.0)
    except ValueError:
        sigma = 2.0 / epsilon
    return sigma


def gaussian_mech_ldp(data, epsilon, delta=1e-5):
    sensitivity = 2.0
    sigma = analytic_gaussian_sigma(epsilon, delta, sensitivity)
    noise = np.random.normal(loc=0.0, scale=sigma, size=data.shape)
    return data + noise


class DataHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.raw_df = None
        self.meta_info = {}

    def load_data(self, columns=None, sample_frac=1.0):

        print(f"Loading {self.file_path}...")
        try:
            if self.file_path.endswith('.parquet'):
                self.raw_df = pd.read_parquet(self.file_path, columns=columns)
            elif self.file_path.endswith('.csv'):
                self.raw_df = pd.read_csv(self.file_path, usecols=columns)

            if sample_frac < 1.0:
                self.raw_df = self.raw_df.sample(frac=sample_frac, random_state=42)

            print(f"Loaded {len(self.raw_df)} rows.")
            return self.raw_df
        except Exception as e:
            print(f"Error loading: {e}")
            return None

    def preprocess_column(self, col_name, method='min-max', clip_quantile=(0.0, 1.0)):

        if col_name not in self.raw_df.columns:
            print(f"Column {col_name} not found.")
            return None


        raw_series = self.raw_df[col_name]


        if raw_series.dtype == 'object':
            print(f"Detected string format in '{col_name}'. Cleaning '$' and ','...")
            raw_series = raw_series.astype(str).str.replace('$', '', regex=False) \
                .str.replace(',', '', regex=False)



        raw_data = pd.to_numeric(raw_series, errors='coerce').values.astype(np.float32)


        valid_mask = ~np.isnan(raw_data)
        data = raw_data[valid_mask]


        dropped_count = len(raw_data) - len(data)
        if dropped_count > 0:
            print(f"Warning: Dropped {dropped_count} rows (NaN or invalid format).")

        if len(data) == 0:
            print("Error: No valid data left after preprocessing!")
            return None


        lower_q, upper_q = clip_quantile


        L = np.percentile(data, lower_q * 100)
        R = np.percentile(data, upper_q * 100)


        if R == L:
            R = L + 1e-5


        clipped_data = np.clip(data, L, R)


        if method == 'abs_max':

            scale = R
            if scale == 0: scale = 1.0
            norm_data = (clipped_data / scale) * 2 - 1


            meta_L, meta_R = 0, R
            shift = 0

        else:
            shift = (L + R) / 2
            scale = (R - L) / 2

            if scale == 0: scale = 1.0  # 防止除零

            norm_data = (clipped_data - shift) / scale
            meta_L, meta_R = L, R


        self.meta_info[col_name] = {
            'L': meta_L,
            'R': meta_R,
            'scale': scale,
            'shift': shift,
            'true_mean_real': np.mean(data)
        }

        print(f"Processed '{col_name}': Real Range [{L:.2f}, {R:.2f}] -> Norm [-1, 1]. Size: {len(norm_data)}")
        return norm_data