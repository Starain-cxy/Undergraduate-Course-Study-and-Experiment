# Mood_Variance_Test    3.3节 Mood方差检验
import numpy as np
from scipy.stats import norm

# ======================== 全局变量与参数设置（请在此处修改） ========================
# 样本数据
A = np.array([4.5, 6.5, 7, 10, 12])
B = np.array([6, 7.2, 8, 9, 9.8])

# 备择假设： "!=" (双侧, 方差不等), ">" (A方差 > B方差), "<" (A方差 < B方差)
H1 = "!="

alpha = 0.05


# ============================== 核心算法函数 ==============================
def mood_test_var(A, B, alternative='!=', alpha=0.05):
    """
    执行Mood方差检验 (严格遵循王星《非参数统计》第二版3.3节定义)

    参数:
        A (np.array): 样本A
        B (np.array): 样本B
        alternative (str): 备择假设
        alpha (float): 显著性水平

    返回:
        dict: 结果字典
    """
    A = np.asarray(A)
    B = np.asarray(B)
    n = len(A)  # 样本A的量
    m = len(B)  # 样本B的量
    N = n + m  # 总量

    # 1. 合并样本并排序，同时记录组别
    # 创建结构化数组: (数值, 组别标记, 原始索引)
    # 组别 0 代表 A，组别 1 代表 B
    combined = np.concatenate([A, B])
    groups = np.array([0] * n + [1] * m)

    # 2. 排序并处理结 (平均秩法)
    # 获取排序索引
    sorted_indices = np.argsort(combined)
    sorted_values = combined[sorted_indices]
    sorted_groups = groups[sorted_indices]

    # 分配秩 (处理结)
    ranks = np.empty(N, dtype=float)
    i = 0
    while i < N:
        end = i
        while end < N and sorted_values[end] == sorted_values[i]:
            end += 1
        # 平均秩
        avg_rank = (i + 1 + end) / 2
        ranks[i:end] = avg_rank
        i = end

    # 3. 提取样本A的秩 (R_i)
    # 找到所有组别为0 (样本A) 的位置对应的秩
    ranks_A = ranks[sorted_groups == 0]

    # 4. 计算 Mood 统计量 M
    # M = sum( (R_i - (N+1)/2 )^2 )
    mid_rank = (N + 1) / 2
    M = np.sum((ranks_A - mid_rank) ** 2)

    # 5. 计算原假设下的期望 E(M) 和方差 Var(M)
    # 教材公式 (3.15)
    E_M = n * (N ** 2 - 1) / 12

    # 教材公式 (3.16) - 无结情况
    Var_M = n * m * (N + 1) * (N ** 2 - 4) / 180

    # 结调整 (如果有结，需要对方差进行调整，教材未详细展开此步，但为严谨通常加上)
    # 计算结统计量
    tie_counts = []
    i = 0
    while i < N:
        end = i
        while end < N and sorted_values[end] == sorted_values[i]:
            end += 1
        tie_counts.append(end - i)
        i = end

    # 只有当存在结且结的长度大于1时才调整
    if any(t > 1 for t in tie_counts):
        # 这是一个更高级的调整，通常大样本下若结不多可忽略，
        # 这里为了保持代码简洁，主要展示教材核心公式，暂不引入复杂调整因子
        # 若需严谨，可参考 Kruskal-Wallis 的结调整方式类比
        pass

    # 6. 计算 Z 统计量 (正态近似)
    Z = (M - E_M) / np.sqrt(Var_M)

    # 7. 确定临界值和 p 值
    # 备择假设逻辑：
    # H1: Var(A) > Var(B) -> M 会很大 -> 拒绝域在右侧 (Z > z_alpha)
    # H1: Var(A) < Var(B) -> M 会很小 -> 拒绝域在左侧 (Z < -z_alpha)

    if alternative == '>':
        p_value = 1 - norm.cdf(Z)
        crit_val = norm.ppf(1 - alpha)
        reject = Z > crit_val
    elif alternative == '<':
        p_value = norm.cdf(Z)
        crit_val = norm.ppf(alpha)
        reject = Z < crit_val
    else:  # '!='
        p_value = 2 * min(norm.cdf(Z), 1 - norm.cdf(Z))
        crit_val_low = norm.ppf(alpha / 2)
        crit_val_high = norm.ppf(1 - alpha / 2)
        reject = p_value < alpha

    return {
        'A': A, 'B': B, 'n': n, 'm': m, 'N': N,
        'sorted_values': sorted_values,
        'sorted_groups': sorted_groups,
        'ranks': ranks,
        'ranks_A': ranks_A,
        'mid_rank': mid_rank,
        'M': M,
        'E_M': E_M,
        'Var_M': Var_M,
        'Z': Z,
        'p_value': p_value,
        'alternative': alternative,
        'alpha': alpha,
        'reject': reject,
        'crit_val': crit_val if alternative != '!=' else None,
        'crit_val_low': crit_val_low if alternative == '!=' else None,
        'crit_val_high': crit_val_high if alternative == '!=' else None
    }


def print_mood_results(res):
    print("\n" + "=" * 70)
    print("              Mood 方差检验 (基于王星《非参数统计》第二版)")
    print("=" * 70)

    alt_desc = {
        '!=': '双侧 (A方差 ≠ B方差)',
        '>': '右侧 (A方差 > B方差)',
        '<': '左侧 (A方差 < B方差)'
    }
    print(f"备择假设 H1 : {res['alternative']} ({alt_desc[res['alternative']]})")
    print(f"显著性水平 α = {res['alpha']}")

    # 打印排序和秩的表格
    print("\n" + "-" * 70)
    print(f"{'混合排序后数据':<15} | {'组别':<6} | {'原始秩':<8} | {'(R_i - (N+1)/2)^2':<20}")
    print("-" * 70)

    mid = res['mid_rank']
    for val, grp, rk in zip(res['sorted_values'], res['sorted_groups'], res['ranks']):
        grp_name = "A" if grp == 0 else "B"
        # 只有样本A参与计算M统计量，标记一下
        contrib = f"{(rk - mid) ** 2:.2f}" if grp == 0 else "-"
        print(f"{val:<15.4f} | {grp_name:<6} | {rk:<8.2f} | {contrib:<20}")

    print("-" * 70)

    # 打印统计量计算
    print(f"\n[1] 样本 A 在混合排序中的秩 R_i: {res['ranks_A']}")
    print(f"[2] 中间秩 (N+1)/2 = {res['mid_rank']}")
    print(f"[3] Mood 统计量 M = Sum[(R_i - (N+1)/2)^2] = {res['M']:.4f}")

    print(f"\n[4] 原假设下的分布:")
    print(f"    期望 E(M) = n(N^2 - 1)/12 = {res['E_M']:.4f}")
    print(f"    方差 Var(M) = nm(N+1)(N^2-4)/180 = {res['Var_M']:.4f}")
    print(f"    Z 统计量 = (M - E(M)) / sqrt(Var(M)) = {res['Z']:.4f}")

    # 决策部分
    print("\n" + "-" * 70)
    print("                      检验决策")
    print("-" * 70)

    if res['alternative'] == '!=':
        print(f"临界值 Z(α/2) = {res['crit_val_low']:.4f}, Z(1-α/2) = {res['crit_val_high']:.4f}")
        print(f"p 值 = {res['p_value']:.6f}")

        # 模仿教材句式
        print(f"\n结论:")
        print(f"Z = {res['Z']:.4f}，在 α={res['alpha']} 水平下，", end="")
        if res['reject']:
            print(f"|Z| > Z(1-α/2) = {res['crit_val_high']:.4f}，故拒绝原假设。")
        else:
            print(f"Z(α/2) = {res['crit_val_low']:.4f} ≤ Z ≤ Z(1-α/2) = {res['crit_val_high']:.4f}，故不能拒绝原假设。")
    else:
        dir_str = "1-α" if res['alternative'] == '>' else "α"
        cv = res['crit_val']
        print(f"临界值 Z({dir_str}) = {cv:.4f}")
        print(f"p 值 = {res['p_value']:.6f}")

        print(f"\n结论:")
        print(f"Z = {res['Z']:.4f}，在 α={res['alpha']} 水平下，", end="")
        if res['reject']:
            if res['alternative'] == '>':
                print(f"Z > Z(1-α) = {cv:.4f}，故拒绝原假设。")
            else:
                print(f"Z < Z(α) = {cv:.4f}，故拒绝原假设。")
        else:
            print(f"故不能拒绝原假设。")

    print("=" * 70)


def main():

    result = mood_test_var(A, B, alternative=H1, alpha=alpha)
    print_mood_results(result)


if __name__ == "__main__":
    main()