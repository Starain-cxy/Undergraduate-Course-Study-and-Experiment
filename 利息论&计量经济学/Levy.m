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
% I2 对角贡献：2/α * h^(-α)
% I3 对角贡献：-2/(2-α) * h^(-α)
% I3 邻点贡献：1/(2-α) * h^(-α)   （用于相邻点）
coef_I2      = 2/alpha * h^(-alpha);
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
M = C_alpha * M;

%% 4. 特征值计算 + 正确尺度归一化
[V, D]   = eig(M);
eig_vals = diag(D);
[lambda_max_d, idx] = max(eig_vals);   % 最大特征值（离散尺度）
lambda_true = lambda_max_d * h^alpha;  % 去量纲后的真实特征值
phi      = V(:, idx);                  % 对应特征向量

% 确保特征向量符号一致（可选）
if phi(1) < 0
    phi = -phi;
end

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
figure('Color','w');
plot(x_d, phi, 'LineWidth', 1.5);
grid on; box on;
xlabel('x_i \in [-1,1]', 'FontSize', 12);
ylabel('\phi(x_i)', 'FontSize', 12);
title('主特征向量 (最大特征值对应)', 'FontSize', 14);