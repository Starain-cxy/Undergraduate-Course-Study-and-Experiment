import random
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def CutbyTime(time, lambd):
    """生成一条泊松过程样本路径，截止时间为 time"""
    t = random.expovariate(lambd)
    series = []
    while t < time:
        series.append(t)
        t += random.expovariate(lambd)
    event = len(series)
    return series, time, event


def chi2_independence_test(samples, T, k_ratio, alpha=0.05, lambd=None):
    """
    对泊松过程在 [0, kT] 和 (kT, T] 两个区间的增量进行卡方独立性检验。

    参数：
    - samples: 长度为 M 的列表，每个元素是一个事件时间列表（已按时间排序，且 < T）
    - T: 总观测时间（浮点数）
    - k_ratio: 分割比例，区间分割点为 a = k_ratio * T，需满足 0 < k_ratio < 1
    - alpha: 显著性水平（默认 0.05）
    - lambd: 泊松过程的真实速率 λ。若为 None，则从数据中估计

    返回值：
    - 字典，包含检验统计量、自由度、p 值、是否拒绝原假设等信息
    """
    if not (0 < k_ratio < 1):
        raise ValueError("k_ratio 必须在 (0, 1) 区间内")

    a = k_ratio * T
    M = len(samples)

    # 步骤1：统计每个样本在 A=[0,a] 和 B=(a,T] 中的事件数
    X = []  # A 区间事件数
    Y = []  # B 区间事件数
    total_events = 0
    for events in samples:
        x = sum(1 for t in events if t <= a)
        y = len(events) - x
        X.append(x)
        Y.append(y)
        total_events += len(events)

    X = np.array(X)
    Y = np.array(Y)

    # 估计 lambda（如果未提供）
    if lambd is None:
        lambd_hat = total_events / (M * T)
        lambda_known = False
    else:
        lambd_hat = lambd
        lambda_known = True

    # 确定合理的最大值（避免过多的空单元格）
    # 使用99.9%分位数或均值+3*标准差，取较小值
    max_x_possible = int(np.ceil(min(np.max(X), np.mean(X) + 3 * np.std(X), np.percentile(X, 99.9))))
    max_y_possible = int(np.ceil(min(np.max(Y), np.mean(Y) + 3 * np.std(Y), np.percentile(Y, 99.9))))

    # 确保至少包含主要的数据点
    max_x_possible = max(max_x_possible, int(np.percentile(X, 95)))
    max_y_possible = max(max_y_possible, int(np.percentile(Y, 95)))

    # 创建列联表，将超出范围的值合并到最后一类
    r = max_x_possible + 1  # 行数（0, 1, ..., max_x_possible）
    c = max_y_possible + 1  # 列数（0, 1, ..., max_y_possible）

    # 观测频数 O_{ij}
    O = np.zeros((r, c), dtype=int)
    for x, y in zip(X, Y):
        # 将超出范围的值合并到边界
        x_idx = min(x, max_x_possible)
        y_idx = min(y, max_y_possible)
        O[x_idx, y_idx] += 1

    # 计算理论期望频数
    if lambda_known:
        # 已知λ：使用泊松分布的理论概率
        # 计算区间A和B的理论泊松参数
        lambda_a = lambd_hat * a
        lambda_b = lambd_hat * (T - a)

        # 计算每个单元格的理论概率
        p_theory = np.zeros((r, c))
        for i in range(r):
            for j in range(c):
                # 对于边界单元格，需要计算累积概率
                if i == max_x_possible or j == max_y_possible:
                    # 计算边界上的累积概率
                    if i == max_x_possible and j == max_y_possible:
                        p_x = 1 - stats.poisson.cdf(max_x_possible - 1, lambda_a)
                        p_y = 1 - stats.poisson.cdf(max_y_possible - 1, lambda_b)
                    elif i == max_x_possible:
                        p_x = 1 - stats.poisson.cdf(max_x_possible - 1, lambda_a)
                        p_y = stats.poisson.pmf(j, lambda_b)
                    else:  # j == max_y_possible
                        p_x = stats.poisson.pmf(i, lambda_a)
                        p_y = 1 - stats.poisson.cdf(max_y_possible - 1, lambda_b)
                else:
                    p_x = stats.poisson.pmf(i, lambda_a)
                    p_y = stats.poisson.pmf(j, lambda_b)
                p_theory[i, j] = p_x * p_y  # 独立性假设

        # 确保概率和为1（处理数值误差）
        p_theory = p_theory / p_theory.sum()
        E = M * p_theory
        df = (r - 1) * (c - 1)

    else:
        # 未知λ：使用数据估计的边缘分布
        # 计算观测的边缘频率
        p_x_obs = np.zeros(r)
        p_y_obs = np.zeros(c)

        for i in range(r):
            p_x_obs[i] = np.sum(O[i, :]) / M

        for j in range(c):
            p_y_obs[j] = np.sum(O[:, j]) / M

        # 在独立性假设下，理论频率是边缘频率的乘积
        E = M * np.outer(p_x_obs, p_y_obs)
        df = (r - 1) * (c - 1) - 1  # 减去一个估计的参数

    # 检查期望频数
    low_expected = np.sum(E < 5)
    if low_expected > 0:
        print(f"警告：有 {low_expected} 个单元格的期望频数小于5，卡方近似可能不可靠。")
        print(f"期望频数的最小值：{E.min():.4f}")

    # 计算卡方统计量（只计算期望频数>0的单元格）
    valid_cells = E > 0
    chi2_stat = np.sum(((O[valid_cells] - E[valid_cells]) ** 2) / E[valid_cells])

    # 计算p值
    p_value = 1 - stats.chi2.cdf(chi2_stat, df)

    # 计算临界值
    critical_val = stats.chi2.ppf(1 - alpha, df)

    # 判断是否拒绝原假设
    reject_H0 = chi2_stat > critical_val

    # 打印结果
    print(f"=== 卡方独立增量性检验 ===")
    print(f"分割点 a = {a:.3f}（k_ratio = {k_ratio}）")
    print(f"样本数量 M = {M}")
    print(f"λ {'已知' if lambda_known else '未知（从数据估计）'}：λ = {lambd_hat:.4f}")
    print(f"列联表大小：{r} × {c}")
    print(f"卡方统计量：{chi2_stat:.4f}")
    print(f"自由度：{df}")
    print(f"临界值（α={alpha}）：{critical_val:.4f}")
    print(f"P 值：{p_value:.6f}")
    if reject_H0:
        print("=> 拒绝原假设 H₀：增量不独立，不符合泊松过程。")
    else:
        print("=> 不拒绝原假设 H₀：无足够证据表明增量不独立。")

    return {
        '卡方统计量': chi2_stat,
        '自由度': df,
        'P值': p_value,
        '拒绝原假设': reject_H0,
        '临界值': critical_val,
        '使用的λ': lambd_hat,
        '列联表形状': (r, c),
        '期望频数最小值': E.min()
    }


def generate_samples(M, T, lambd_true):
    """生成 M 条泊松过程样本路径"""
    samples = []
    for _ in range(M):
        series, _, _ = CutbyTime(T, lambd_true)
        samples.append(series)
    return samples


def plot_event_counts(samples, T, k_ratio):
    """绘制两个区间事件数的联合频数热力图"""
    a = k_ratio * T
    X, Y = [], []
    for events in samples:
        x = sum(1 for t in events if t <= a)
        y = len(events) - x
        X.append(x)
        Y.append(y)

    max_x, max_y = max(X), max(Y)
    joint = np.zeros((max_x + 1, max_y + 1), dtype=int)
    for x, y in zip(X, Y):
        joint[x, y] += 1

    plt.figure(figsize=(8, 6))
    sns.heatmap(joint, annot=True, fmt="d", cmap="Blues", cbar_kws={'label': '频数'})
    plt.title(f'[0, {a:.2f}] 与 ({a:.2f}, {T}] 区间事件数联合分布')
    plt.xlabel(f'({a:.2f}, {T}] 区间事件数')
    plt.ylabel(f'[0, {a:.2f}] 区间事件数')
    plt.tight_layout()
    plt.show()


# -----------------------------
# 示例使用
# -----------------------------

if __name__ == "__main__":
    # 设置随机种子以保证结果可复现
    random.seed(42)
    np.random.seed(42)

    # 参数设置
    M = 2000
    T = 5.0
    lambd_true = 1.5  # 适中的λ值
    k_ratio = 0.5  # 使用中点分割
    alpha = 0.05  # 显著性水平

    # 生成样本
    samples = generate_samples(M, T, lambd_true)

    # 情况1：λ 未知（从数据估计）
    print("【检验1】λ 未知（从数据中估计）")
    result1 = chi2_independence_test(samples, T, k_ratio, alpha=alpha, lambd=None)

    print("\n" + "=" * 60 + "\n")

    # 情况2：λ 已知（提供真实值）
    print("【检验2】λ 已知（使用真实值）")
    result2 = chi2_independence_test(samples, T, k_ratio, alpha=alpha, lambd=lambd_true)

    # 可视化联合分布
    plot_event_counts(samples, T, k_ratio)