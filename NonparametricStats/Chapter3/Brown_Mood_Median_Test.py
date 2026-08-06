# Brown_Mood_Median_Test    3.1节 Brown_Mood 中位数检验
import numpy as np
from scipy.stats import hypergeom, norm

# ======================== 全局变量与参数设置 ========================
A = np.array([698, 688, 675, 656, 655, 648, 640, 639, 620])
B = np.array([780, 754, 740, 712, 693, 680, 621])

# 备择假设： "!=" (双侧), ">" (A中位数 > B中位数), "<" (A中位数 < B中位数)
H1 = "<"

alpha = 0.05
use_correction = True  # 正态近似时是否使用连续性校正


# ============================== 核心算法函数 ==============================
def brown_mood_median_test(A, B, alternative='!=', alpha=0.05, correction=True):
    A = np.asarray(A)
    B = np.asarray(B)
    n1, n2 = len(A), len(B)
    N = n1 + n2

    # 1. 计算合并中位数
    combined = np.concatenate([A, B])
    median_combined = np.median(combined)

    # 2. 构建列联表 (a 是样本A中 <= 中位数的个数)
    a = np.sum(A <= median_combined)
    b = n1 - a
    c = np.sum(B <= median_combined)
    d = n2 - c

    # 3. 超几何分布参数设定
    # 定义：总体中共有 N 个球，其中 M_col 个是"成功"（<=中位数），
    # 从总体中抽取 n1 个（样本A的量），其中有 a 个成功球。
    M_col = a + c  # 总体中 <= 中位数的总数
    k = a  # 观察到的成功数

    # 超几何分布模型: hypergeom(N, K, n) -> K是总体成功数，n是抽取数
    rv = hypergeom(N, M_col, n1)

    # 4. 计算精确 p 值
    if alternative == '>':
        # H1: A的中位数 > B的中位数 -> 意味着A中 <= M的应该少，即 a 太小。
        # 拒绝域在左侧。P(X <= k)
        p_exact = rv.cdf(k)
    elif alternative == '<':
        # H1: A的中位数 < B的中位数 -> 意味着A中 <= M的应该多，即 a 太大。
        # 拒绝域在右侧。P(X >= k)
        p_exact = rv.sf(k - 1)
    else:
        # 双侧检验：计算所有概率小于等于当前点概率的和
        p_obs = rv.pmf(k)
        # 生成所有可能的 x
        x_min = max(0, n1 - (N - M_col))
        x_max = min(n1, M_col)
        x_values = np.arange(x_min, x_max + 1)
        pmf_values = rv.pmf(x_values)
        p_exact = np.sum(pmf_values[pmf_values <= p_obs + 1e-12])

    # 5. 计算正态近似 (Z 检验)
    # 超几何分布的均值和方差
    mean_hyper = n1 * M_col / N
    var_hyper = (n1 * M_col * (N - M_col) * (N - n1)) / (N ** 2 * (N - 1))
    std_hyper = np.sqrt(var_hyper)

    # 连续性校正
    if correction:
        if alternative == '<':  # 检验右侧 P(X >= k)，校正为 k - 0.5
            z_num = k - 0.5 - mean_hyper
        elif alternative == '>':  # 检验左侧 P(X <= k)，校正为 k + 0.5
            z_num = k + 0.5 - mean_hyper
        else:  # 双侧
            if k > mean_hyper:
                z_num = k - 0.5 - mean_hyper
            elif k < mean_hyper:
                z_num = k + 0.5 - mean_hyper
            else:
                z_num = 0
    else:
        z_num = k - mean_hyper

    Z = z_num / std_hyper

    # 计算 Z 分布的 p 值
    if alternative == '<':
        p_approx = 1 - norm.cdf(Z)  # 右侧概率
    elif alternative == '>':
        p_approx = norm.cdf(Z)  # 左侧概率
    else:
        p_approx = 2 * (1 - norm.cdf(abs(Z)))  # 双侧

    # 决策逻辑：小样本用精确，大样本用近似
    # 这里的阈值可以调整，通常 n1, n2 都大于 20 用近似比较好
    use_exact = (n1 <= 20 and n2 <= 20)
    if use_exact:
        p_value = p_exact
        method = "精确超几何概率"
    else:
        p_value = p_approx
        method = f"正态近似 (Z检验, 校正={correction})"

    reject = p_value < alpha
    conclusion = "拒绝原假设" if reject else "不能拒绝原假设"

    return {
        'median_combined': median_combined,
        'table': np.array([[a, b], [c, d]]),
        'n1': n1, 'n2': n2, 'N': N,
        'k': k, 'mean_hyper': mean_hyper, 'var_hyper': var_hyper,
        'Z': Z,
        'p_approx': p_approx,
        'p_exact': p_exact,
        'p_value': p_value,
        'method': method,
        'reject': reject,
        'conclusion': conclusion,
        'alpha': alpha,
        'alternative': alternative,
        'use_exact': use_exact,
        'correction': correction
    }


def print_results(res):
    print("\n" + "=" * 60)
    print("           Brown-Mood 中位数检验")
    print("=" * 60)

    alt_desc = {'!=': '双侧 (A中位数 ≠ B中位数)',
                '>': '右侧 (A中位数 > B中位数)',
                '<': '左侧 (A中位数 < B中位数)'}

    print(f"备择假设 H1 : {res['alternative']} ({alt_desc[res['alternative']]})")
    print(f"显著性水平 α = {res['alpha']}")
    print(f"合并中位数 M = {res['median_combined']:.4f}")

    print(f"\n[ 列联表 ]")
    print(f"          ≤ M    > M    合计")
    print(f"样本 A    {res['table'][0, 0]:^4d}   {res['table'][0, 1]:^4d}   {res['n1']:^4d}")
    print(f"样本 B    {res['table'][1, 0]:^4d}   {res['table'][1, 1]:^4d}   {res['n2']:^4d}")
    print(
        f"合计      {res['table'][0, 0] + res['table'][1, 0]:^4d}   {res['table'][0, 1] + res['table'][1, 1]:^4d}   {res['N']:^4d}")

    print(f"\n[ 精确检验 (超几何分布) ]")
    print(f"观察统计量 a (样本A中≤M的数量) = {res['k']}")
    print(f"精确 p 值 = {res['p_exact']:.6f}")

    print(f"\n[ 近似检验 (正态分布) ]")
    print(f"期望均值 E(a) = {res['mean_hyper']:.4f}")
    print(f"方差 Var(a)   = {res['var_hyper']:.4f}")
    print(f"Z 统计量      = {res['Z']:.4f}")
    print(f"近似 p 值     = {res['p_approx']:.6f}")

    print(f"\n[ 最终结论 ]")
    print(f"采用方法: {res['method']}")
    print(f"p 值 = {res['p_value']:.6f}")
    if res['reject']:
        print(f"结论: 拒绝原假设 (p < {res['alpha']})，认为两样本中位数存在显著差异。")
    else:
        print(f"结论: 不能拒绝原假设 (p ≥ {res['alpha']})，尚无足够证据表明两样本中位数不同。")


def main():
    result = brown_mood_median_test(A, B, alternative=H1, alpha=alpha, correction=use_correction)
    print_results(result)


if __name__ == "__main__":
    main()