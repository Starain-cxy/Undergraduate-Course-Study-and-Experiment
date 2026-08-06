import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import random
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def poisson_goodness_of_fit_test(event, time, alpha=0.05, lambda_param=None):
    """
    泊松分布卡方拟合优度检验

    参数:
    ----------
    event_counts : list or np.array
        记录每个样本中事件数的数组
    sample_time : float
        截断时间 T
    alpha : float
        显著性水平，默认 0.05
    lambda_param : float or None
        已知的 λ 值，如果为 None 则用样本均值估计
    """
    M = len(event)  # 样本数量
    print(f"样本数量 M = {M}")
    print(f"截断时间 T = {time}")


    unique_counts, observed_freq = np.unique(event, return_counts=True)
    K = len(unique_counts)

    if lambda_param is not None:
        # λ 已知的情况
        print(f"   使用已知参数 λ = {lambda_param:.4f}")
        λ = lambda_param
    else:
        # λ 未知，用样本均值估计
        λ = np.mean(event) / time
        print(f"   用样本均值估计 λ = {λ:.4f}")

    μ = λ * time  # 泊松分布参数 μ = λT
    print(f"   泊松分布参数 μ = λT = {μ:.4f}")

    # 3. 计算理论概率
    theoretical_probs = stats.poisson.pmf(unique_counts, μ)
    theoretical_freq = M * theoretical_probs

    # 4. 计算卡方统计量
    chi2_stat = 0
    print(f"\n2. 卡方统计量计算:")

    for i in range(K):
        term = (observed_freq[i] - theoretical_freq[i]) ** 2 / theoretical_freq[i]
        chi2_stat += term


    print(f"\n   卡方统计量 χ² = {chi2_stat:.4f}")

    if lambda_param is not None:
        # λ 已知，自由度为 K
        df = K
        critical_value = stats.chi2.ppf(1 - alpha, df)
    else:
        # λ 未知，自由度为 K-1（估计了一个参数）
        df = K - 1
        critical_value = stats.chi2.ppf(1 - alpha, df)

    print(f"   自由度 df = {df}")
    print(f"   显著性水平 α = {alpha}")
    print(f"   临界值 χ²_{1 - alpha:.3f}({df}) = {critical_value:.4f}")
    print(f"   拒绝域: [χ² ≥ {critical_value:.4f}]")

    # 6. 做出决策
    p_value = 1 - stats.chi2.cdf(chi2_stat, df)

    print(f"   χ² = {chi2_stat:.4f}, p-value = {p_value:.6f}")

    if chi2_stat >= critical_value:
        print(f"   ✗ 在显著性水平 {alpha} 下拒绝原假设 H0")
        print(f"   即认为事件数不服从泊松分布 Pois({μ:.4f})")
    else:
        print(f"   ✓ 在显著性水平 {alpha} 下不拒绝原假设 H0")
        print(f"   即没有足够证据表明事件数不服从泊松分布 Pois({μ:.4f})")

    return {
        'chi2_stat': chi2_stat,
        'df': df,
        'p_value': p_value,
        'critical_value': critical_value,
        'reject': chi2_stat >= critical_value,
        'lambda_hat': λ if lambda_param is None else lambda_param,
        'mu': μ,
        'unique_counts': unique_counts,
        'observed_freq': observed_freq,
        'theoretical_freq': theoretical_freq
    }


def plot_poisson_distribution(event_counts, sample_time, lambda_param=None):
    """
    绘制事件数分布图
    """
    if lambda_param is None:
        λ = np.mean(event_counts) / sample_time
    else:
        λ = lambda_param

    μ = λ * sample_time

    plt.figure(figsize=(8, 5))

    # 右图：观测频数 vs 理论频数
    unique_counts, observed_freq = np.unique(event_counts, return_counts=True)
    theoretical_freq = len(event_counts) * stats.poisson.pmf(unique_counts, μ)

    x_pos = np.arange(len(unique_counts))
    width = 0.35

    plt.bar(x_pos - width / 2, observed_freq, width, label='观测频数', alpha=0.7)
    plt.bar(x_pos + width / 2, theoretical_freq, width, label='理论频数', alpha=0.7)

    plt.xlabel('事件数')
    plt.ylabel('频数')
    plt.title('观测频数与理论频数对比')
    plt.xticks(x_pos, unique_counts)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def generate_poisson_samples(num_samples=5000, sample_time=100.0, lambda_true=0.02):
    """
    生成泊松过程样本

    参数:
    ----------
    num_samples : int
        样本数量
    sample_time : float
        截断时间 T
    lambda_true : float
        真实速率 λ
    """
    print(f"生成 {num_samples} 个泊松过程样本:")
    print(f"  截断时间 T = {sample_time}")
    print(f"  真实速率 λ = {lambda_true}")
    print(f"  期望事件数 μ = λT = {lambda_true * sample_time}")

    # 生成样本
    event_counts = []

    for i in range(num_samples):
        # 使用给定的 CutbyTime 函数
        _, _, event = CutbyTime(sample_time, lambda_true)
        event_counts.append(event)

        # 显示进度
        if (i + 1) % 1000 == 0:
            print(f"  已生成 {i + 1}/{num_samples} 个样本")

    return np.array(event_counts)


def test_poisson_goodness_of_fit():
    """
    生成样本并进行泊松分布检验
    """
    print("=" * 60)
    print("泊松过程检验实验")
    print("=" * 60)

    # 参数设置
    sample_time = 100.0  # 截断时间 T
    lambda_true = 0.02  # 真实速率 λ
    num_samples = 5000  # 样本数量

    # 1. 生成样本
    event_counts = generate_poisson_samples(num_samples, sample_time, lambda_true)

    print(f"\n样本统计信息:")
    print(f"  事件数均值: {np.mean(event_counts):.4f}")
    print(f"  事件数方差: {np.var(event_counts):.4f}")
    print(f"  均值/方差比: {np.mean(event_counts) / np.var(event_counts):.4f} (泊松分布应接近1)")

    # 2. 绘制分布图
    plot_poisson_distribution(event_counts, sample_time, lambda_true)

    print("\n" + "=" * 60)
    print("情况1: λ 未知，用样本均值估计")
    print("=" * 60)

    # 情况1: λ 未知
    result1 = poisson_goodness_of_fit_test(
        event=event_counts,
        time=sample_time,
        alpha=0.05,
        lambda_param=None
    )

    print("\n" + "=" * 60)
    print("情况2: λ 已知（使用真实值）")
    print("=" * 60)

    # 情况2: λ 已知
    result2 = poisson_goodness_of_fit_test(
        event=event_counts,
        time=sample_time,
        alpha=0.05,
        lambda_param=lambda_true
    )

    return event_counts, result1, result2


# 定义 CutbyTime 函数（使用你提供的版本）
def CutbyTime(time, lambd):
    """生成泊松过程的事件时间序列"""
    t = random.expovariate(lambd)
    series = []
    while t < time:
        series.append(t)
        random_number = random.expovariate(lambd)
        t = t + random_number
    event = len(series)
    return series, time, event


# 使用示例
if __name__ == "__main__":
    # 设置随机种子保证结果可重复
    np.random.seed(42)
    random.seed(42)

    # 运行检验
    event_counts, result1, result2 = test_poisson_goodness_of_fit()