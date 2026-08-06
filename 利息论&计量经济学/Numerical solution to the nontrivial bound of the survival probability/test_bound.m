%% 对称α-稳定过程 首达概率与特征函数（全α绘制 + 强制非负特征函数）
clear; clc; close all;

%% 1. 参数设置
N = 1000;                % 网格点数
x = linspace(-1,1,N);   % 离散节点 x
alpha_list = 0.1:0.05:1.9; % alpha：0.1~1.9，步长0.1（共19个）
t = 1;                  % 时间
R = 1;                  % 区间半径

%% 2. 预分配存储
lambda1_list = zeros(size(alpha_list));
upper_bound_list = zeros(size(alpha_list));
phi1_all = zeros(N, length(alpha_list));

%% 3. 循环计算每个alpha
for idx = 1:length(alpha_list)
    alpha = alpha_list(idx);
    fprintf('正在计算 alpha = %.2f ...\n', alpha);
    
    % 系数
    c_alpha = gamma(1+alpha) * sin(alpha*pi/2) / pi;
    h = 2/N;
    
    % 构造 Toeplitz 矩阵 V
    V = zeros(N);
    diag_val = 2 * c_alpha * (zeta(1+alpha) - 1) / (h^alpha);
    offdiag_val = @(d) -c_alpha / (h^alpha) / (d+1)^(1+alpha);
    
    for p = 1:N
        for q = 1:N
            if p == q
                V(p,q) = diag_val;
            else
                d = abs(p - q);
                V(p,q) = offdiag_val(d);
            end
        end
    end
    
    % 求最小实特征值 + 特征向量（第一特征函数）
    [Vec, Val] = eigs(V, 1, 'smallestreal');
    lambda1 = Val(1);
    phi1 = Vec(:,1);
    
    % ===================== 关键修改 =====================
    % 强制特征函数为非负数（符合物理意义：概率/密度非负）
    % 如果整体为负，翻转符号；如果有正有负，取绝对值保证全正
    % ====================================================
    phi1 = abs(phi1);       % 强制全部为正数
    phi1 = phi1 / norm(phi1);% 归一化
    
    % 保存
    lambda1_list(idx) = lambda1;
    phi1_all(:,idx) = phi1;
    
    % 首达概率上界
    upper_bound = exp(-lambda1 * t);
    upper_bound_list(idx) = upper_bound;
end

%% 4. 绘制所有 α 的第一特征函数（19条全部画出）
figure('Name','所有 α 对应的第一特征函数');
hold on; grid on; box on;
cmap = parula(length(alpha_list)); % 彩色区分不同alpha

for idx = 1:length(alpha_list)
    plot(x, phi1_all(:,idx), 'LineWidth',1.5, 'Color',cmap(idx,:), ...
         'DisplayName',sprintf('α=%.1f',alpha_list(idx)));
end

xlabel('x'); ylabel('\phi_1(x) （第一特征函数）');
title('对称α-稳定过程：所有 α 对应的第一特征函数');
legend('Location','best');
colormap(cmap);

%% 5. 绘制首达概率上界随 α 变化曲线
figure('Name','首达概率上界');
hold on; grid on; box on;
plot(alpha_list, upper_bound_list, 'k-o', 'LineWidth',2.5, 'MarkerSize',2);
xlabel('\alpha'); ylabel('首达概率上界');
title('首达概率上界随 \alpha 变化曲线');
xlim([0.1,1.9]);

%% 6. 输出结果
fprintf('\n===== 计算结果 =====\n');
fprintf('alpha\t第一特征值\t上界\n');
for i = 1:length(alpha_list)
    fprintf('%.1f\t%.4f\t%.4f\n', alpha_list(i), lambda1_list(i), upper_bound_list(i));
end