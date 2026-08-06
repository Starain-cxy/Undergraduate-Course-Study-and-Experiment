import numpy as np
from scipy import stats


def generate_poisson_samples(n_samples, t, lam):
    mu = lam * t
    return np.random.poisson(mu, n_samples)


def cut_by_time(t, lam):
    series = []
    current_t = np.random.exponential(1 / lam)
    while current_t < t:
        series.append(current_t)
        current_t += np.random.exponential(1 / lam)
    return len(series)


def poisson_gof_test(counts, t, alpha=0.05, lam_given=None):
    """泊松拟合优度检验（修正了稀疏组合并问题）"""
    M = len(counts)

    # 1. 统计观测频数
    unique_obs, freq_obs = np.unique(counts, return_counts=True)

    # 估计参数或使用给定参数
    if lam_given is None:
        lam_hat = np.mean(counts) / t
    else:
        lam_hat = lam_given

    mu = lam_hat * t

    # 2. 计算理论概率与频数
    prob_theory = stats.poisson.pmf(unique_obs, mu)
    freq_theory = M * prob_theory

    # 3. 关键修正：合并理论频数小于5的尾部组
    # 这是人类写统计代码时必须做的，否则卡方检验无效
    min_expected = 5
    while np.any(freq_theory < min_expected):
        # 找到理论频数最小的位置并合并到最后一个组
        idx = np.argmin(freq_theory)
        if idx == len(freq_theory) - 1:
            break
        # 合并观测值和频数
        freq_obs[idx:-1] += freq_obs[-1]
        freq_obs = freq_obs[:-1]

        freq_theory[idx:-1] += freq_theory[-1]
        freq_theory = freq_theory[:-1]
        unique_obs = unique_obs[:-1]

        # 重新计算概率（防止数值误差）
        prob_theory = stats.poisson.pmf(unique_obs, mu)
        freq_theory = M * prob_theory

    # 4. 计算卡方统计量
    chi2_stat = np.sum((freq_obs - freq_theory) ** 2 / freq_theory)

    # 确定自由度
    # 如果参数未知，自由度需减去估计参数的个数 (k-1-1)
    # 这里简化处理，通常为 k-2 (k组，估计了lambda)
    df = len(unique_obs) - 2 if lam_given is None else len(unique_obs)

    p_value = 1 - stats.chi2.cdf(chi2_stat, df)
    critical = stats.chi2.ppf(1 - alpha, df)

    # 输出结果
    print(f"样本量: {M}, 截断时间: {t}")
    print(f"估计/给定 λ: {lam_hat:.4f}, 对应 μ: {mu:.4f}")
    print(f"卡方统计量: {chi2_stat:.4f}, 自由度: {df}")
    print(f"p值: {p_value:.6f}, 临界值: {critical:.4f}")

    if chi2_stat > critical:
        print("✗ 拒绝原假设 (不服从泊松分布)")
    else:
        print("✓ 不拒绝原假设 (符合泊松分布)")

    return chi2_stat, p_value, df


if __name__ == "__main__":
    # 设置随机种子以保证可重复性
    np.random.seed(55)

    # 参数
    T = 100.0
    LAM_TRUE = 0.02
    N_SAMPLES = 5000

    # 生成数据（使用高效的向量化方法）
    events = generate_poisson_samples(N_SAMPLES, T, LAM_TRUE)

    print("=== 情况1: λ未知 (需估计) ===")
    poisson_gof_test(events, T, alpha=0.05, lam_given=None)

    print("\n=== 情况2: λ已知 (真实值) ===")
    poisson_gof_test(events, T, alpha=0.05, lam_given=LAM_TRUE)