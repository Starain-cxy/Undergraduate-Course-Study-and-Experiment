%% 对称α-稳定过程：首达概率上界随实际阈值变化
clear; clc; close all;

%% ========== 用户输入区 ==========
alpha_fit = 1.221455;   % 从 Python 拟合的 α
scale_fit = 0.662621;   % 从 Python 拟合的尺度参数
t = 1;                % 时间（天）
R_target = 0.3;      % 您特别关注的阈值（例如 3%），用于标记
% =================================

%% 1. 计算第一特征值 λ1(alpha_fit)
N = 1000;              % 网格点数
alpha = alpha_fit;
c_alpha = gamma(1+alpha) * sin(alpha*pi/2) / pi;
h = 2 / N;
% 构造 Toeplitz 矩阵
diag_val = 2 * c_alpha * (zeta(1+alpha) - 1) / (h^alpha);
V = zeros(N);
for p = 1:N
    for q = 1:N
        if p == q
            V(p,q) = diag_val;
        else
            d = abs(p-q);
            V(p,q) = -c_alpha / (h^alpha) / (d+1)^(1+alpha);
        end
    end
end
[~, val] = eigs(V, 1, 'smallestreal');
lambda1 = real(val);
fprintf('α = %.4f 时，第一特征值 λ1 = %.6f\n', alpha_fit, lambda1);

%% 2. 定义实际阈值范围（从接近0到合理上限）
R_min = 0.01;    % 最小阈值（避免除以0）
R_max = 10;     % 最大阈值（20%）
R_vec = linspace(R_min, R_max, 1500);   % 200个点

% 计算每个阈值下的上界
upper_bound = zeros(size(R_vec));
for i = 1:length(R_vec)
    R = R_vec(i);
    R_eff = R / scale_fit;   % 归一化半径
    upper_bound(i) = exp(-lambda1 * (R_eff^(-alpha_fit)) * t);
end

%% 3. 计算特定阈值 R_target 的上界（用于标记）
R_eff_target = R_target / scale_fit;
upper_target = exp(-lambda1 * (R_eff_target^(-alpha_fit)) * t);
fprintf('阈值 R=%.4f 对应的首达概率上界 = %.6f\n', R_target, upper_target);

%% 4. 绘图
figure;
plot(R_vec, upper_bound, 'b-', 'LineWidth', 2); hold on;
plot(R_target, upper_target, 'ro', 'MarkerSize', 10, 'LineWidth', 2);

% 添加标注
text(R_target, upper_target, ...
    sprintf(' R=%.3f\n 上界=%.4f', R_target, upper_target), ...
    'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', 'FontSize', 10);

xlabel('实际阈值 R（收益率波动绝对值）');
ylabel('首达概率上界 P_0(\tau_R > t)');
title(sprintf('对称α-稳定过程首达概率上界 (α=%.3f, scale=%.4f, t=%.1f天)', ...
              alpha_fit, scale_fit, t));
grid on;
xlim([0, R_max]);
ylim([0, 1]);
legend('上界曲线', sprintf('R=%.3f 标记', R_target), 'Location','best');