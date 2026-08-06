# Lévy过程首达概率的数值估计

‍

# 计算特征值推导

取步长 $h = \frac{2}{n}$，不妨设 $n$ 为偶数。

令 $x_k = - 1 + hk$ ($k = 1, 2 \dots n - 1$)，$x_0 =-1, x_n = 1$，其中 $x_{\frac{n}{2}} = 0$。

令 $E = \mathbb{R} \backslash ( - h , h )$，这里暂时取 $C_\alpha = 1$。

则算子 $\mathcal{A}$ 作用于函数 $\phi(x)$ 的积分表达式为：

$$
\begin{aligned}
	\mathcal{A}\phi(x) & =\int_{\mathbb{R}\setminus\{0\}}\left(\phi(x+z)-\phi(x)\right)\frac{dz}{|z|^{1+\alpha}}                                                                                 \\
	                   & =\int_{E}\phi(x+z)\frac{dz}{|z|^{1+\alpha}}-\phi(x)\int_{E}\frac{dz}{|z|^{1+\alpha}}+\int_{(-h,h)\setminus\{0\}}\left(\phi(x+z)-\phi(x)\right)\frac{dz}{|z|^{1+\alpha}}
\end{aligned}
$$

将上述积分分为三部分 $I_1, I_2, I_3$，给定 $x = x_i$：

## 1. 第一部分积分 $I_1$

对 $I_1$ 考虑区域 $- 1 < x_i + z < 1$，即 $x_i + z$ 遍历网格点 $x_{1} , x_{2} \cdots x_{i - 1} , x_{i + 1} \cdots x_{n - 1}$：

$$
I_{1} = \sum_{k = 1 , k \neq i}^{n - 1} \phi ( x_{k} ) \frac{h}{| x_{k} - x_{i} |^{1 + \alpha}}
$$

## 2. 第二部分积分 $I_2$

对 $I_2$ 进行计算：$-\phi(x_i) \cdot 2\int_h^\infty \frac{dz}{z^{1+\alpha}}$

$$
I_{2}= -\phi ( x_{i}) \left( \int_{-\infty}^{-h}\frac{d z}{| z |^{1 + \alpha}}+ \int
_{h}^{\infty}\frac{d z}{| z |^{1 + \alpha}}\right)
$$

$$
= -\phi ( x_{i}) \cdot 2 \int_{h}^{+\infty}\frac{d z}{z^{1 + \alpha}}= -\frac{2}{\alpha}
h^{-\alpha}\phi ( x_{i})
$$

## 3. 第三部分积分 $I_3$

对 $I_3$ 在 $(-h, h) \setminus \{0\}$ 中考虑泰勒展开：

$$
\phi(x_{i} + z) - \phi(x_{i}) = \phi'(x_{i})z + \frac{1}{2}\phi''(x_{i})z^{2} + o
(z^{3})
$$

在积分中，第一项为奇函数积分为 0，则：

$$
I_{3} \approx \int_{0}^{h} \phi''(x_i) z^{2} \frac{d z}{z^{1 + \alpha}} = \phi''(x_i) \int_{0}^{h} z^{1 - \alpha} d z = \frac{1}{(2-\alpha)} \phi''(x_i) h^{2 - \alpha}
$$

而 $\phi''(x_i)$ 近似为二阶差分：

$$
\phi''(x_{i}) \approx \frac{1}{h^{2}}( \phi ( x_{i + 1}) - 2 \phi ( x_{i}) + \phi
( x_{i - 1}) )
$$

则：

$$
I_3 \approx \frac{1}{2 - \alpha} h^{-\alpha} ( \phi ( x_{i + 1} ) - 2 \phi ( x_{i} ) + \phi ( x_{i - 1} ) )
$$

## 4. 综合结果

综上所述，离散化后的算子 $A$ 作用于 $\phi(x_i)$ 为：

$$
\mathcal{A}\phi(x_{i})=\sum_{k=1,k\neq i}^{n-1}\phi(x_{k})\frac{h}{| x_{k}- x_{i}|^{1 + \alpha}}-\frac{2}{\alpha}h^{-\alpha}\phi(x_{i})+\frac{1}{2 - \alpha}h^{-\alpha}(\phi(x_{i+1})-2\phi(x_{i})+\phi(x_{i-1}))
$$

选取 $( \phi ( x_{1} ) , \phi ( x_{2} ) \cdots \phi ( x_{n - 1} ) )$ 的系数，有：

$$
M_{ij}=
\begin{cases}
	-\left( \frac{2}{\alpha}+ \frac{2}{2-\alpha}\right) h^{-\alpha}=-\frac{4}{\alpha \left(2-\alpha\right)}h^{-\alpha} & j = i                 \\
	\left( \frac{1}{2-\alpha}+ 1 \right) h^{-\alpha}                                                                   & j = i+1 \text{ 或 }i-1 \\
	\frac{h}{|x_j - x_i|^{1+\alpha}}                                                                                   & \text{其它 }j
\end{cases}
$$

令 $M = ( M_{i i} )_{( n - 1 ) \times ( n - 1 )}$ 为系数矩阵，$v = ( \phi ( x_{1} ) , \phi ( x_{2} ) \cdots \phi ( x_{n - 1} ) )^{T}$ 为特征向量。

则特征值问题 $\mathcal{A} \phi ( x ) = \lambda \phi ( x)$ 转化为矩阵特征值问题：

$$
M v = \lambda v
$$

MATLAB 对矩阵运算支持较好，用其中已经支持的函数求矩阵最大特征值及其对应特征向量 $v' = ( \phi' ( x_{1} ) \cdots \phi' ( x_{n-1} ) )$ 并绘制 $( x_{i} , \phi' ( x_{i} ) )$ 图像。

‍

‍

# 理论补充

首达时间 $\tau_\epsilon = \inf\{t > 0 : |Z_t| \ge \epsilon\}$,  
从 $Z_0 = x$ 出发，首达概率 $P_x(\tau_\epsilon > t) = u(x, t)$  $x \in (-\epsilon, \epsilon)$  
满足向后方程：

$$
\frac{\partial u}{\partial t} = \mathcal{A} u \quad , \quad u(x, 0) = 1
$$

解形式为 $u(x, t) = e^{\lambda t} \phi(x)$，方程变为 $\mathcal{A} \phi(x) = \lambda \phi(x)$

由微分方程解的结构，

$$
u(x, t) = \sum_{i} c_{i} \phi(x) \exp\{\lambda_{i} t\}
$$

考虑 $u(x, 0) = 1$ 则 $\sum_i c_i \phi(x) = 1$，则 $P(\tau_\epsilon > t) = \sum_i c_i \phi(x) \exp\{\lambda_i t\}$

取 $\lambda_{1}= \lambda_{\min}$，则 $P(\tau_\epsilon > t) \ge e^{\lambda_1 t} \sum_i c_i \phi(x) = e^{\lambda_1 t}$，$t=1$时：

$$
P(\sup_{0 < s < 1} |Z_s| < \epsilon) = P(\tau_\epsilon > 1) = e^{\lambda_1 t}
$$

对于不同 $\epsilon$，$\mathcal{A} \phi_\epsilon(x) = \lambda_\epsilon \phi(x)$ 的特征向量相同

$\lambda_1$ 是 $\epsilon$ 的函数，$\lambda_1(\epsilon) = \epsilon \lambda_1(1)$，所以只要解 $\epsilon = 1$ 时的 $\lambda_1 = \lambda_1(1)$，就有

$$
P(\sup_{0 < s < 1}|Z_{s}| < \epsilon) \ge e^{\lambda_1 \epsilon}
$$

注意 $\lambda_1$ 是 $\mathcal{A}$ 的最小特征值！

$\mu_1$ 为 $-\mathcal{A}$ 的最大特征值，有 $\mu_1 = -\lambda_1$

$$
P(\sup_{0 < s < 1} |Z_s| < \epsilon) \ge e^{-\mu_1 \epsilon}
$$

查阅资料，Lévy 过程的固有参数为

$$
\boldsymbol{C_\alpha = \frac{\alpha \cdot \Gamma(\alpha)}{\pi} \sin\left( \frac{\pi \alpha}{2} \right)}
$$

所以我们实际上需要解特征值问题：

$$
-C_{\alpha}M v = \lambda v
$$

# 概率下界的合理保证

概率的下界显然不能大于1，即$-C_{\alpha}M$的特征值最大值不能小于0，通过**不可约对角占优矩阵的正定性判定定理**可以保证这一点。

## 1. 矩阵定义

给定 $\alpha \in (1,2)$，步长 $h = \frac{2}{n}$（$n$ 偶数），节点 $x_k = -1 + hk$，$k = 0,1,\dots,n$，其中 $x_0=-1,\;x_n=1$，内部节点 $k=1,\dots,n-1$。矩阵 $M \in \mathbb{R}^{(n-1)\times (n-1)}$ 的元素为

$$
M_{ij}=
\begin{cases}
-\left( \frac{2}{\alpha}+ \frac{2}{2-\alpha}\right) h^{-\alpha}=-\frac{4}{\alpha \left(2-\alpha\right)}h^{-\alpha}, & j = i \\[6pt]
\left( \frac{1}{2-\alpha}+ 1 \right) h^{-\alpha}, & j = i+1 \text{ 或 }i-1 \\[6pt]
\dfrac{h}{|x_j - x_i|^{1+\alpha}}, & \text{其它 }j .
\end{cases}
$$

考虑矩阵 $-M$。其元素为

$$
(-M)_{ii} = \frac{4}{\alpha(2-\alpha)}h^{-\alpha} > 0,\qquad
(-M)_{ij} = -\frac{h}{|x_j-x_i|^{1+\alpha}} < 0\;(i\neq j).
$$

## 2. 不可约对角占优 M‑矩阵定理

**定理 (Horn & Johnson, *Matrix Analysis*, Corollary 6.2.27)**   
设 $A = (a_{ij})$ 是实对称矩阵，满足：

1. $a_{ii} > 0$ 对所有 $i$；
2. $a_{ij} \le 0$ 对所有 $i \neq j$；
3. $A$ 是不可约的（即其关联图强连通）；
4. 至少存在一个下标 $i_0$ 使得 $a_{i_0 i_0} > \sum_{j \neq i_0} |a_{i_0 j}|$（不可约对角占优）。

则 $A$ 是正定矩阵（所有特征值 > 0）。  
（这样的矩阵称为 **不可约对角占优 M‑矩阵**，其逆矩阵元素全为正，且所有特征值具有正实部；由于对称性，特征值全为正实数。）

![image](assets/image-20260402000922-88hdiyg.png)

## 3. 验证条件

### 3.1 正对角元与负非对角元

由定义，$(-M)_{ii} = \frac{4}{\alpha(2-\alpha)}h^{-\alpha} > 0$，且对于 $i\neq j$，$(-M)_{ij} = -\frac{h}{|x_j-x_i|^{1+\alpha}} < 0$。条件 1、2 成立。

### 3.2 不可约性

节点集合 $\{1,2,\dots,n-1\}$ 中，每个节点 $i$ 与所有其他节点 $j$ 都有非零元（因为 $M_{ij}$ 对任意 $i\neq j$ 均非零），因此矩阵的图是完全图，显然不可约。条件 3 成立。

### 3.3 严格对角占优行

取边界节点 $i=1$（或 $i=n-1$）。该行的非对角元绝对值之和为

$$
\sum_{j\neq 1}|(-M)_{1j}| = |(-M)_{12}| + \sum_{j=3}^{n-1} |(-M)_{1j}|.
$$

计算：

$$
|(-M)_{12}| = \left(\frac{1}{2-\alpha}+1\right)h^{-\alpha} = \frac{3-\alpha}{2-\alpha}h^{-\alpha},
$$

$$
|(-M)_{1j}| = \frac{h}{|x_j-x_1|^{1+\alpha}} = \frac{1}{(j-1)^{1+\alpha}}h^{-\alpha},\quad j\ge 3.
$$

令 $S_n = \sum_{k=2}^{n-2} \frac{1}{k^{1+\alpha}}$，则

$$
(-M)_{11} - \sum_{j\neq 1}|(-M)_{1j}| = h^{-\alpha}\left[ \frac{4}{\alpha(2-\alpha)} - \frac{3-\alpha}{2-\alpha} - S_n \right].
$$

化简括号内的前两项：

$$
\frac{4}{\alpha(2-\alpha)}- \frac{3-\alpha}{2-\alpha}= \frac{4 - \alpha(3-\alpha)}{\alpha(2-\alpha)}
= \frac{\alpha^{2} - 3\alpha + 4}{\alpha(2-\alpha)}.
$$

对于 $\alpha\in(1,2)$，$\alpha^2-3\alpha+4 = (\alpha-1.5)^2+1.75 > 0$，分母 $\alpha(2-\alpha) > 0$，故该表达式为正。而且

$$
\sum_{k=2}^{\infty} \frac{1}{k^{1+\alpha}} \le \sum_{k=2}^{\infty} \frac{1}{k^{2}} = \frac{\pi^2}{6} - 1 \approx 0.6449.
$$

另一方面，函数 $f(\alpha)=\frac{\alpha^2-3\alpha+4}{\alpha(2-\alpha)}$ 在 $\alpha\in(1,2)$ 上的最小值出现在 $\alpha=1$ 或 $\alpha=2$ 的极限。计算：

$$
\lim_{\alpha\to 1^+} f(\alpha) = \frac{1-3+4}{1\cdot 1} = 2,\quad
\lim_{\alpha\to 2^-} f(\alpha) = \frac{4-6+4}{2\cdot 0} \to +\infty.
$$

因此 $f(\alpha) > 2$ 对 $\alpha\in(1,2)$ 成立，而 $\sum_{k=2}^{\infty} \frac{1}{k^{1+\alpha}} \le 0.6449 < 2$。于是对任意 $n$ 有

$$
\frac{4}{\alpha(2-\alpha)} - \frac{3-\alpha}{2-\alpha} - S_n > 2 - 0.6449 > 0.
$$

故 $(-M)_{11} > \sum_{j\neq 1}|(-M)_{1j}|$，即第一行严格对角占优。条件 4 成立。

![image](assets/image-20260401233457-szy6e3a.png)

## 4. 结论

矩阵 $-M$ 满足不可约对角占优 M‑矩阵的所有条件，因此 $-M$ 是正定矩阵。特别地，其所有特征值均为正实数，从而最大特征值 $\lambda_{\max}(-M) > 0$。

‍

# 数值方法的收敛性

模拟计算结果发现特征值异常大，且数量级随步长 $h$ 增大而增大，学习到数值计算存在 **“分数阶导数的离散化尺度”** ，应对计算得来的特征值乘一个 $h^{\alpha}$ 去量纲处理，因为对于离散化求解的$\lambda_{\text{disc}}(h)$与连续的真实特征值$\lambda_{\text{cont}}$存在关系

$$
\lambda_{\text{cont}}= \lim_{h\to 0}\big( \lambda_{\text{disc}}(h) \cdot h^{\alpha}
\big)
$$

## 代码实现

MATLAB 对矩阵运算支持较好，用其中已经支持的函数求矩阵最大特征值及其对应特征向量 $v' = ( \phi' ( x_{1} ) \cdots \phi' ( x_{n-1} ) )$ 并绘制 $( x_{i} , \phi' ( x_{i} ) )$ 图像。

按上述方法，计算特征值，特征向量和概率下界的估计

$$
P\left( \sup_{0 < s \leq 1} |Z_s| < \varepsilon \right) > e^{-\lambda\varepsilon}
$$

计算结果发现特征值异常大，且数量级随步长 $h$ 增大而增大，学习到数值计算时应对计算得来的特征值除一个 $h^{-\alpha}$ 去量纲处理，根据上证指数数据估计得到 $\alpha \approx 1.46$，带入这个值和 $\varepsilon=0.1$ 得到：

```plaintext
========== 计算结果 ==========
n      = 1000
alpha  = 1.4600
eps    = 0.1000
h      = 2.00e-03
lambda_true = 3.243226
P > exp(-λ·ε) = 0.72301697
==============================
```

‍

‍

#### *具体代码

```matlab
%% 对称α-稳定Lévy过程 首达概率下界（正确尺度归一化 + 绘图）
% 公式：P( sup_{0<s≤1} |Z_s| < ε ) > exp(-λ·ε)
% 正确尺度：λ_true = 离散特征值 / h^alpha
clear; clc; close all;

%% ===================== 可调参数 =====================
n       = 1000;        % 网格点数（偶数）
alpha   = 1.46;        % 稳定过程阶数 0<alpha<2
epsilon = 0.1;         % 阈值 ε
%% ====================================================

%% 1. 对称 α-稳定过程标准系数 C_α
C_alpha = (alpha * gamma(alpha)/pi) * sin(pi*alpha/2);

%% 2. 网格
h       = 2 / n;                  % 步长
x_full  = -1 : h : 1;             % 全网格（包含边界）
x_d     = x_full(2:end-1);        % 内部点（Dirichlet 边界）
m       = length(x_d);            % 内部点数

%% 3. 构造离散生成元矩阵 M
M = zeros(m, m);

% ------------------- 非局部项 I1 -------------------
% 对所有 j ≠ i，贡献为 h / |x_j - x_i|^(1+alpha)
for i = 1:m
    for j = 1:m
        if i ~= j
            M(i,j) = h / abs(x_d(j) - x_d(i))^(1+alpha);
        end
    end
end

% ------------------- 局部项 I2 + I3 -------------------
% I2 对角贡献：-2/α * h^(-α)
% I3 对角贡献：-2/(2-α) * h^(-α)
% I3 邻点贡献：1/(2-α) * h^(-α)   （用于相邻点）
coef_I2      = -2/alpha * h^(-alpha);
coef_I3_diag = -2/(2-alpha) * h^(-alpha);
coef_I3_adj  = 1/(2-alpha) * h^(-alpha);

for i = 1:m
    % 对角元：已有的非局部项 + I2 + I3 对角
    M(i,i) = M(i,i) + coef_I2 + coef_I3_diag;
    % 邻点：已有的非局部项 + I3 邻点补充
    if i > 1
        M(i,i-1) = M(i,i-1) + coef_I3_adj;
    end
    if i < m
        M(i,i+1) = M(i,i+1) + coef_I3_adj;
    end
end

% 乘以常数 C_α
M = -C_alpha * M;

%% 4. 特征值计算 + 正确尺度归一化
[V, D]   = eig(M);
eig_vals = diag(D);
[lambda_max_d, idx] = max(eig_vals);   % 最大特征值（离散尺度）
lambda_true = lambda_max_d * h^alpha;  % 去量纲后的真实特征值
phi      = V(:, idx);                  % 对应特征向量

% 确保特征向量符号一致（可选）
% if phi(1) < 0
%     phi = abs(phi);
% end
% phi=abs(phi)

%% 5. 概率下界
prob_lower = exp(-lambda_true * epsilon);

%% 6. 输出结果
fprintf('\n========== 计算结果 ==========\n');
fprintf('n      = %d\n', n);
fprintf('alpha  = %.4f\n', alpha);
fprintf('eps    = %.4f\n', epsilon);
fprintf('h      = %.2e\n', h);
fprintf('lambda_true = %.6f\n', lambda_true);
fprintf('P > exp(-λ·ε) = %.8f\n', prob_lower);
fprintf('==============================\n');

%% 7. 绘图：主特征向量
figure;
scatter(x_d, phi, 10, 'filled');  % 散点图，看清离散点
```
