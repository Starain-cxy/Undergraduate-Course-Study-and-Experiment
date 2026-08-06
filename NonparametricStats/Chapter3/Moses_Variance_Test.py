# Moses_Variance_Test    3.4节 Moses方差检验

import numpy as np
from scipy.stats import norm
import warnings

# ======================== 全局变量与参数设置 ========================
A = np.array([8.2, 10.7, 7.5, 14.6, 6.3, 9.2, 11.9, 5.6, 12.8, 5.2, 4.9, 13.5])
B = np.array([4.7, 6.3, 5.2, 6.8, 5.6, 4.2, 6.0, 7.4, 8.1, 6.5])

k = 3                     # 每组观测值个数
H1 = "!="                 # "!=", ">", "<"
alpha = 0.05
random_seed = 123
verbose = True

# ============================== 核心算法函数 ==============================
def moses_variance_test(A, B, k, alternative='!=', alpha=0.05, seed=None, verbose=True):
    A = np.asarray(A).copy()
    B = np.asarray(B).copy()
    n_A, n_B = len(A), len(B)

    # 1. 确定分组数，丢弃多余观测
    m1 = n_A // k
    m2 = n_B // k
    if n_A % k != 0 or n_B % k != 0:
        warnings.warn(
            f"样本量不是 k={k} 的整数倍。"
            f"A 组舍弃最后 {n_A % k} 个观测，B 组舍弃最后 {n_B % k} 个观测。"
        )

    rng = np.random.RandomState(seed)
    A_trim = rng.permutation(A[:m1 * k])
    B_trim = rng.permutation(B[:m2 * k])

    # 2. 分成 k 个一组的小组
    subgroups_A = A_trim.reshape(m1, k)
    subgroups_B = B_trim.reshape(m2, k)

    # 3. 计算各小组离均差平方和
    def subgroup_ss(subgroup):
        return np.sum((subgroup - np.mean(subgroup)) ** 2)

    SS_A = np.apply_along_axis(subgroup_ss, 1, subgroups_A)
    SS_B = np.apply_along_axis(subgroup_ss, 1, subgroups_B)

    # 4. 混合编秩
    SS_all = np.concatenate([SS_A, SS_B])
    n_total = m1 + m2
    ranks = np.zeros(n_total)
    sorted_idx = np.argsort(SS_all)
    sorted_SS = SS_all[sorted_idx]

    i = 0
    while i < n_total:
        j = i
        while j < n_total and np.isclose(sorted_SS[j], sorted_SS[i]):
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[sorted_idx[i:j]] = avg_rank
        i = j

    ranks_A = ranks[:m1]
    ranks_B = ranks[m1:]

    # 5. 计算两组的秩和
    S_A = np.sum(ranks_A)
    S_B = np.sum(ranks_B)

    # 6. 根据备择假设选择 S 与对应分组数（关键修正）
    if alternative == '!=':
        # 双侧检验：取较小秩和为 S（与教材例题一致）
        if S_A <= S_B:
            S = S_A
            m_used = m1
            side_used = 'A'
        else:
            S = S_B
            m_used = m2
            side_used = 'B'
    elif alternative == '>':
        S = S_A
        m_used = m1
        side_used = 'A'
    else:  # '<'
        S = S_B
        m_used = m2
        side_used = 'B'

    T_M = S - m_used * (m_used + 1) / 2.0

    # 7. 正态近似检验（期望与方差基于较小 m 或指定 m）
    #    注：期望与方差公式中的 m, n 应使用对应的分组数
    m_small = min(m1, m2)
    m_large = max(m1, m2)
    # 对于双侧，较小秩和对应的分组数就是 m_used，另一组为 m_other
    m_other = m1 if m_used == m2 else m2
    EU = m_used * m_other / 2.0
    VarU = m_used * m_other * (m_used + m_other + 1) / 12.0

    # 连续性校正与 Z 值
    if alternative == '!=':
        if T_M <= EU:
            Z = (T_M + 0.5 - EU) / np.sqrt(VarU)
        else:
            Z = (T_M - 0.5 - EU) / np.sqrt(VarU)
        p_value = 2 * norm.cdf(-abs(Z))
    elif alternative == '>':
        Z = (T_M - 0.5 - EU) / np.sqrt(VarU)
        p_value = norm.sf(Z)
    else:  # '<'
        Z = (T_M + 0.5 - EU) / np.sqrt(VarU)
        p_value = norm.cdf(Z)

    # 临界值
    if alternative == '!=':
        z_crit_low = norm.ppf(alpha / 2)
        z_crit_high = norm.ppf(1 - alpha / 2)
        crit_low = EU + z_crit_low * np.sqrt(VarU)
        crit_high = EU + z_crit_high * np.sqrt(VarU)
    elif alternative == '>':
        z_crit = norm.ppf(1 - alpha)
        crit_low = None
        crit_high = EU + z_crit * np.sqrt(VarU)
    else:
        z_crit = norm.ppf(alpha)
        crit_low = EU + z_crit * np.sqrt(VarU)
        crit_high = None

    reject = p_value < alpha
    conclusion = "拒绝原假设" if reject else "不能拒绝原假设"

    result = {
        'k': k, 'm1': m1, 'm2': m2,
        'SS_A': SS_A, 'SS_B': SS_B,
        'ranks_A': ranks_A, 'ranks_B': ranks_B,
        'S_A': S_A, 'S_B': S_B,
        'S_used': S, 'm_used': m_used, 'side_used': side_used,
        'T_M': T_M,
        'EU': EU, 'VarU': VarU, 'Z': Z, 'p_value': p_value,
        'crit_low': crit_low, 'crit_high': crit_high,
        'reject': reject, 'conclusion': conclusion,
        'alpha': alpha, 'alternative': alternative,
        'subgroups_A': subgroups_A, 'subgroups_B': subgroups_B
    }

    if verbose:
        _print_moses_result(result)

    return result


def _print_moses_result(res):
    print("\n" + "=" * 70)
    print("                Moses 方差检验（王星《非参数统计》3.4节）")
    print("=" * 70)
    print(f"每组观测值数 k = {res['k']}")
    print(f"A 样本分组数 m1 = {res['m1']}  |  B 样本分组数 m2 = {res['m2']}")
    print(f"显著性水平 α = {res['alpha']}  |  备择假设 H1 : {res['alternative']}")
    print("-" * 70)

    print("\n【样本 A 各小组离差平方和及秩】")
    for i in range(res['m1']):
        sg = np.round(res['subgroups_A'][i], 3)
        print(f"  组 {i+1:2d}: 数据 {sg}  →  SS = {res['SS_A'][i]:.4f}  秩 = {res['ranks_A'][i]:.1f}")

    print("\n【样本 B 各小组离差平方和及秩】")
    for i in range(res['m2']):
        sg = np.round(res['subgroups_B'][i], 3)
        print(f"  组 {i+1:2d}: 数据 {sg}  →  SS = {res['SS_B'][i]:.4f}  秩 = {res['ranks_B'][i]:.1f}")

    print("\n" + "-" * 70)
    print(f"样本 A 的秩和 S_A = {res['S_A']:.1f}")
    print(f"样本 B 的秩和 S_B = {res['S_B']:.1f}")
    print(f"双侧检验采用较小秩和 S = min(S_A, S_B) = {res['S_used']:.1f} (来自样本 {res['side_used']})")
    print(f"Moses 统计量 T_M = S - m_used*(m_used+1)/2 = {res['T_M']:.4f}")
    print(f"Mann-Whitney U 期望 E(U) = {res['EU']:.2f}  |  方差 Var(U) = {res['VarU']:.4f}")
    print(f"正态近似 Z 统计量 = {res['Z']:.4f}")
    print(f"近似 p 值 = {res['p_value']:.4f}")

    if res['alternative'] == '!=':
        print(f"双侧临界值 (近似) : U_L = {res['crit_low']:.2f}, U_H = {res['crit_high']:.2f}")
    elif res['alternative'] == '>':
        print(f"单侧临界值 (近似) : U_crit = {res['crit_high']:.2f}")
    else:
        print(f"单侧临界值 (近似) : U_crit = {res['crit_low']:.2f}")

    print("\n" + "=" * 70)
    print("结论：")
    print(f"  p = {res['p_value']:.4f}  {'<' if res['reject'] else '≥'}  α = {res['alpha']}")
    print(f"  → {res['conclusion']}，认为两总体方差"
          f"{'存在显著差异' if res['reject'] else '无显著差异'}。")
    print("=" * 70)
    print("注：本结果基于正态近似，若 m1, m2 较小，请查教材附表 4 精确判断。\n")


def main():

    moses_variance_test(A, B, k, alternative=H1, alpha=alpha, seed=random_seed, verbose=True)


if __name__ == "__main__":
    main()