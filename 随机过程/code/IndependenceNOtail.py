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


def chi2_independence_test(samples, T, k_ratio, alpha=0.05, lambd=None, keep_ratio=0.99):
    """
    对泊松过程在 [0, kT] 和 (kT, T] 两个区间的增量进行卡方独立性检验。

    参数：
    - samples: 长度为 M 的列表，每个元素是一个事件时间列表（已按时间排序，且 < T）
    - T: 总观测时间（浮点数）
    - k_ratio: 分割比例，区间分割点为 a = k_ratio * T，需满足 0 < k_ratio < 1
    - alpha: 显著性水平（默认 0.05）
    - lambd: 泊松过程的真实速率 λ。若为 None，则从数据中估计
    - keep_ratio: 保留数据的比例，超出此范围的数据将被丢弃（默认0.99）

    返回值：
    - 字典，包含检验统计量、自由度、p 值、是否拒绝原假设等信息
    """
    if not (0 < k_ratio < 1):
        raise ValueError("k_ratio 必须在 (0, 1) 区间内")

    if keep_ratio <= 0 or keep_ratio > 1:
        raise ValueError("keep_ratio 必须在 (0, 1] 区间内")

    a = k_ratio * T

    # 步骤1：统计每个样本在 A=[0,a] 和 B=(a,T] 中的事件数
    X = []  # A 区间事件数
    Y = []  # B 区间事件数
    total_events = 0
    valid_samples = []  # 有效样本

    for events in samples:
        x = sum(1 for t in events if t <= a)
        y = len(events) - x
        X.append(x)
        Y.append(y)
        total_events += len(events)

    X = np.array(X)
    Y = np.array(Y)

    # 步骤2：确定K值并过滤超出范围的数据
    # 使用分位数确定K值（保留keep_ratio比例的数据）
    K_X = int(np.percentile(X, keep_ratio * 100))
    K_Y = int(np.percentile(Y, keep_ratio * 100))

    print(f"初始样本数: {len(samples)}")
    print(f"基于 {keep_ratio * 100:.1f}% 分位数确定: K_X={K_X}, K_Y={K_Y}")

    # 过滤超出范围的数据
    mask = (X <= K_X) & (Y <= K_Y)
    X_filtered = X[mask]
    Y_filtered = Y[mask]
    M_filtered = len(X_filtered)

    # 记录被丢弃的样本数
    discarded = len(samples) - M_filtered
    discard_rate = discarded / len(samples) * 100

    print(f"丢弃超出范围的样本数: {discarded}/{len(samples)} ({discard_rate:.2f}%)")
    print(f"最终用于检验的样本数: {M_filtered}")

    if M_filtered < 30:
        raise ValueError(f"有效样本数过少 ({M_filtered})，无法进行可靠的卡方检验")

    # 步骤3：重新计算统计量
    total_events_filtered = X_filtered.sum() + Y_filtered.sum()

    # 估计 lambda（如果未提供）
    if lambd is None:
        lambd_hat = total_events_filtered / (M_filtered * T)
        lambda_known = False
    else:
        lambd_hat = lambd
        lambda_known = True

    # 步骤4：构建列联表
    r = K_X + 1  # 行数 (0, 1, ..., K_X)
    c = K_Y + 1  # 列数 (0, 1, ..., K_Y)

    O = np.zeros((r, c), dtype=int)  # 观测频数

    for x, y in zip(X_filtered, Y_filtered):
        O[x, y] += 1

    # 步骤5：计算期望频数
    if lambda_known:
        # 已知λ：使用泊松分布理论
        lambda_a = lambd_hat * a
        lambda_b = lambd_hat * (T - a)

        # 计算理论概率
        p_theory = np.zeros((r, c))
        for i in range(r):
            for j in range(c):
                p_x = stats.poisson.pmf(i, lambda_a)
                p_y = stats.poisson.pmf(j, lambda_b)
                p_theory[i, j] = p_x * p_y

        # 归一化（处理截断）
        p_theory = p_theory / p_theory.sum()
        E = M_filtered * p_theory
        df = (r - 1) * (c - 1)

    else:
        # 未知λ：使用观测的边缘分布
        # 计算观测边缘频率
        p_x_obs = np.sum(O, axis=1) / M_filtered  # 行边缘概率
        p_y_obs = np.sum(O, axis=0) / M_filtered  # 列边缘概率

        # 独立性假设下的理论概率
        p_theory = np.outer(p_x_obs, p_y_obs)
        E = M_filtered * p_theory
        df = (r - 1) * (c - 1) - 1  # 减去一个估计参数

    # 步骤6：检查期望频数
    low_expected = np.sum(E < 5)
    if low_expected > 0:
        print(f"警告：有 {low_expected}/{r * c} 个单元格的期望频数小于5")
        print(f"期望频数范围: [{E.min():.4f}, {E.max():.4f}]")

    # 步骤7：计算卡方统计量
    chi2_stat = 0
    for i in range(r):
        for j in range(c):
            if E[i, j] > 0:  # 避免除以0
                chi2_stat += ((O[i, j] - E[i, j]) ** 2) / E[i, j]

    # 步骤8：计算p值和临界值
    p_value = 1 - stats.chi2.cdf(chi2_stat, df)
    critical_val = stats.chi2.ppf(1 - alpha, df)

    # 步骤9：判断是否拒绝原假设
    reject_H0 = chi2_stat > critical_val

    # 步骤10：打印结果
    print(f"\n=== 卡方独立增量性检验 ===")
    print(f"分割点 a = {a:.3f}（k_ratio = {k_ratio}）")
    print(f"原始样本数: {len(samples)}，有效样本数: {M_filtered}")
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
        '有效样本数': M_filtered,
        '丢弃样本数': discarded,
        '列联表形状': (r, c),
        '期望频数最小值': E.min() if E.size > 0 else 0
    }


def generate_samples(M, T, lambd_true):
    """生成 M 条泊松过程样本路径"""
    samples = []
    for _ in range(M):
        series, _, _ = CutbyTime(T, lambd_true)
        samples.append(series)
    return samples


def plot_event_counts(samples, T, k_ratio, keep_ratio=0.99):
    """绘制两个区间事件数的联合频数热力图"""
    a = k_ratio * T
    X, Y = [], []

    # 统计事件数
    for events in samples:
        x = sum(1 for t in events if t <= a)
        y = len(events) - x
        X.append(x)
        Y.append(y)

    X = np.array(X)
    Y = np.array(Y)

    # 确定K值（与检验中相同）
    K_X = int(np.percentile(X, keep_ratio * 100))
    K_Y = int(np.percentile(Y, keep_ratio * 100))

    # 创建列联表
    joint = np.zeros((K_X + 1, K_Y + 1), dtype=int)

    for x, y in zip(X, Y):
        if x <= K_X and y <= K_Y:  # 只包含有效数据
            joint[x, y] += 1

    plt.figure(figsize=(10, 8))
    sns.heatmap(joint, annot=True, fmt="d", cmap="Blues", cbar_kws={'label': '频数'})
    plt.title(
        f'[0, {a:.2f}] 与 ({a:.2f}, {T}] 区间事件数联合分布\n(保留{keep_ratio * 100:.0f}%数据，K_X={K_X}, K_Y={K_Y})')
    plt.xlabel(f'({a:.2f}, {T}] 区间事件数')
    plt.ylabel(f'[0, {a:.2f}] 区间事件数')
    plt.tight_layout()
    plt.show()

    # 打印统计数据
    print(f"\n=== 数据分布统计 ===")
    print(f"X (区间A事件数): 均值={X.mean():.2f}, 标准差={X.std():.2f}, 最大值={X.max()}")
    print(f"Y (区间B事件数): 均值={Y.mean():.2f}, 标准差={Y.std():.2f}, 最大值={Y.max()}")
    print(f"确定的K值: K_X={K_X}, K_Y={K_Y}")
    print(f"超出范围的数据比例: {np.sum((X > K_X) | (Y > K_Y)) / len(X) * 100:.2f}%")

    return joint


# -----------------------------
# 示例使用
# -----------------------------

if __name__ == "__main__":
    # 设置随机种子以保证结果可复现
    random.seed(42)
    np.random.seed(42)

    # 参数设置
    M = 2000  # 样本路径数量
    T = 5.0  # 总观测时间
    lambd_true = 1.5  # 真实速率参数 λ
    k_ratio = 0.1  # 分割比例（中点分割）
    alpha = 0.05  # 显著性水平
    keep_ratio = 0.99  # 保留99%的数据

    print("参数设置:")
    print(f"  M = {M} (样本数)")
    print(f"  T = {T} (观测时间)")
    print(f"  λ = {lambd_true} (真实速率)")
    print(f"  k_ratio = {k_ratio} (分割比例)")
    print(f"  alpha = {alpha} (显著性水平)")
    print(f"  keep_ratio = {keep_ratio} (数据保留比例)\n")

    # 生成样本
    print("正在生成样本...")
    samples = generate_samples(M, T, lambd_true)
    print(f"生成 {len(samples)} 条样本路径\n")

    # 情况1：λ 未知（从数据估计）
    print("=" * 60)
    print("【检验1】λ 未知（从数据中估计）")
    result1 = chi2_independence_test(samples, T, k_ratio, alpha=alpha,
                                     lambd=None, keep_ratio=keep_ratio)

    print("\n" + "=" * 60 + "\n")

    # 情况2：λ 已知（提供真实值）
    print("【检验2】λ 已知（使用真实值）")
    result2 = chi2_independence_test(samples, T, k_ratio, alpha=alpha,
                                     lambd=lambd_true, keep_ratio=keep_ratio)

    # 可视化联合分布
    print("\n" + "=" * 60)
    print("生成数据分布热力图...")
    joint_matrix = plot_event_counts(samples, T, k_ratio, keep_ratio)

    # 打印总结
    print("\n" + "=" * 60)
    print("检验结果总结:")
    print(f"1. λ未知时的检验:")
    print(f"   卡方统计量: {result1['卡方统计量']:.4f}")
    print(f"   P值: {result1['P值']:.6f}")
    print(f"   结论: {'拒绝' if result1['拒绝原假设'] else '不拒绝'} H₀")

    print(f"\n2. λ已知时的检验:")
    print(f"   卡方统计量: {result2['卡方统计量']:.4f}")
    print(f"   P值: {result2['P值']:.6f}")
    print(f"   结论: {'拒绝' if result2['拒绝原假设'] else '不拒绝'} H₀")