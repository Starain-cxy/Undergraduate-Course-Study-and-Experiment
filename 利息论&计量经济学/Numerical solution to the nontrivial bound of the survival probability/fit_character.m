clear; clc; close all;

% ================== 1. 加载数据 ==================
load('log_returns.mat', 'data');   % data 为列向量，对数收益率
returns = data(:);
n = length(returns);

fprintf('========== 描述性统计 ==========\n');
fprintf('样本量: %d\n', n);
fprintf('均值: %.6f\n', mean(returns));
fprintf('标准差: %.6f\n', std(returns));
fprintf('偏度: %.4f\n', skewness(returns));
fprintf('超额峰度: %.4f\n', kurtosis(returns)-3);
fprintf('\n');

% ================== 2. 特征函数法拟合 ==================
fprintf('========== 特征函数法拟合 (对称稳定, β=0, loc=0) ==========\n');

% 经验特征函数实部（向量化）
emp_cf_real = @(t) mean(cos(t * returns'), 2);  % t为列向量，返回列向量

% 理论特征函数实部
theo_cf_real = @(t, alpha, sigma) exp(- (sigma * abs(t)) .^ alpha);

% 积分上限 T（截尾，取使特征函数衰减到足够小）
T = 25.0;  

% 目标函数：积分 [0, T] (emp - theo)^2 dt
obj_fun = @(params) integral(@(t) (emp_cf_real(t) - theo_cf_real(t, params(1), params(2))).^2, ...
                             0, T, 'ArrayValued', true);

% 初始值
init_alpha = 1.5;
init_sigma = std(returns);
fprintf('初始值: alpha=%.4f, sigma=%.6f\n', init_alpha, init_sigma);

% 优化（有边界）
options = optimoptions('fmincon', 'Display', 'iter', 'Algorithm', 'sqp');
lb = [0.1; 1e-6];
ub = [1.99; Inf];
params0 = [init_alpha; init_sigma];
params = fmincon(obj_fun, params0, [], [], [], [], lb, ub, [], options);
alpha_cf = params(1);
sigma_cf = params(2);
fprintf('特征函数法拟合结果: alpha=%.6f, sigma=%.6f\n', alpha_cf, sigma_cf);

% ================== 3. 卡方拟合优度检验（修正版：等概率分组） ==================
fprintf('\n========== 卡方检验（等概率分组） ==========\n');

% 分组数 k，建议满足 n/k >= 5
k = 10;
if n/k < 5
    warning('每组期望频数不足5，建议减少分组数');
end

% 生成等概率分组边界（基于理论分布分位数）
prob_edges = (0:k)/k;   % 0, 0.1, 0.2, ..., 1.0
edges = arrayfun(@(p) stable_quantile(p, alpha_cf, sigma_cf), prob_edges);

% 实际频数
obs_counts = histcounts(returns, edges);

% 理论频数：等概率分组下每组概率为 1/k，期望频数恒为 n/k
exp_prob = 1/k;
exp_counts = ones(1, k) * n * exp_prob;

% 计算卡方统计量
chi2_stat = sum((obs_counts - exp_counts).^2 ./ exp_counts);

% 自由度：组数 - 估计参数个数 - 1
r = 2;  % 估计了 alpha 和 sigma 两个参数
df = k - r - 1;

if df <= 0
    error('自由度小于等于0，请增加分组数或减少估计参数');
end

p_val = 1 - chi2cdf(chi2_stat, df);

fprintf('分组数: %d, 自由度: %d\n', k, df);
fprintf('卡方统计量 = %.4f\n', chi2_stat);
fprintf('p值 = %.4f\n', p_val);
alpha_sig = 0.05;
crit = chi2inv(1 - alpha_sig, df);
fprintf('显著性水平 %.2f 下的临界值 = %.4f\n', alpha_sig, crit);
if chi2_stat > crit
    fprintf('拒绝 H0: 收益率不服从该对称稳定分布\n');
else
    fprintf('不能拒绝 H0: 收益率服从该对称稳定分布\n');
end

% ================== 4. 可视化 ==================
% 4.1 特征函数拟合图
t_plot = linspace(0.01, 5.0, 300);
emp_plot = emp_cf_real(t_plot');
theo_plot = theo_cf_real(t_plot', alpha_cf, sigma_cf);
figure;
plot(t_plot, emp_plot, 'b-', 'LineWidth', 1.5); hold on;
plot(t_plot, theo_plot, 'r--', 'LineWidth', 1.5);
xlabel('t');
ylabel('Re{φ(t)}');
title(sprintf('特征函数拟合 (α=%.2f, σ=%.4f)', alpha_cf, sigma_cf));
legend('经验特征函数', '拟合特征函数');
grid on;

% 4.2 直方图与理论密度
x_dense = linspace(quantile(returns,0.01), quantile(returns,0.99), 500);
cdf_dense = arrayfun(@(x) cdf_stable_scalar(x, alpha_cf, sigma_cf), x_dense);
h = x_dense(2) - x_dense(1);
pdf_dense = zeros(size(x_dense));
pdf_dense(2:end-1) = (cdf_dense(3:end) - cdf_dense(1:end-2)) / (2*h);
pdf_dense(1) = (cdf_dense(2) - cdf_dense(1)) / h;
pdf_dense(end) = (cdf_dense(end) - cdf_dense(end-1)) / h;
pdf_dense = max(0, pdf_dense);

figure;
histogram(returns, 60, 'Normalization', 'pdf', 'FaceColor', [0.8 0.8 1], 'EdgeColor', 'none');
hold on;
plot(x_dense, pdf_dense, 'r-', 'LineWidth', 2);
xlabel('对数收益率');
ylabel('概率密度');
title('经验直方图 vs 拟合密度');
legend('经验', '拟合稳定分布');
grid on;

% 4.3 CDF 对比
sorted_ret = sort(returns);
ecdf_vals = (1:n)'/n;
cdf_fit = arrayfun(@(x) cdf_stable_scalar(x, alpha_cf, sigma_cf), sorted_ret);
figure;
plot(sorted_ret, ecdf_vals, 'b-', 'LineWidth', 1.5); hold on;
plot(sorted_ret, cdf_fit, 'r--', 'LineWidth', 1.5);
xlabel('对数收益率');
ylabel('累积概率');
title('经验CDF vs 拟合CDF');
legend('经验CDF', '拟合CDF');
grid on;

% 4.4 卡方检验分组频数对比
figure;
bar_width = 0.35;
x_pos = 1:k;
bar(x_pos - bar_width/2, obs_counts, bar_width, 'FaceColor', [0.2 0.6 0.8], 'EdgeColor', 'k');
hold on;
bar(x_pos + bar_width/2, exp_counts, bar_width, 'FaceColor', [0.9 0.6 0.2], 'EdgeColor', 'k');
set(gca, 'XTick', x_pos, 'XTickLabel', arrayfun(@(i) sprintf('G%d', i), 1:k, 'UniformOutput', false));
xlabel('分组');
ylabel('频数');
title('卡方检验：观察频数 vs 期望频数 (等概率分组)');
legend('观察频数', '期望频数');
grid on;

% ================== 5. 输出参数 ==================
fprintf('\n=========================================\n');
fprintf('拟合参数: alpha = %.6f, sigma = %.6f\n', alpha_cf, sigma_cf);
fprintf('卡方检验 p值 = %.4f\n', p_val);
fprintf('=========================================\n');

% ================== 局部函数定义（文件末尾） ==================
% 修正后的单点CDF：符号已更正
function F = cdf_stable_scalar(x, alpha, sigma)
    if x == 0
        F = 0.5;
        return;
    end
    % 对称性：x<0 时 F(x) = 1 - F(|x|)
    if x < 0
        F = 1 - cdf_stable_scalar(-x, alpha, sigma);
        return;
    end
    % x>0 时：正确公式为 0.5 + 积分项
    integrand = @(t) (sin(t*x) ./ t) .* exp(- (sigma^alpha) * (t.^alpha));
    I = integral(integrand, 0, Inf, 'RelTol', 1e-8, 'AbsTol', 1e-10);
    F = 0.5 + (1/pi) * I;
    % 数值截断到 [0,1]
    F = max(0, min(1, F));
end

% 二分法求稳定分布分位数
function x = stable_quantile(p, alpha, sigma)
    if p <= 0
        x = -Inf;
        return;
    end
    if p >= 1
        x = Inf;
        return;
    end
    if p == 0.5
        x = 0;
        return;
    end
    % 利用对称性
    if p < 0.5
        x = -stable_quantile(1-p, alpha, sigma);
        return;
    end
    
    % 二分查找上界
    hi = sigma;
    while cdf_stable_scalar(hi, alpha, sigma) < p
        hi = hi * 2;
    end
    lo = 0;
    
    % 二分迭代
    for iter = 1:100
        mid = (lo + hi) / 2;
        Fmid = cdf_stable_scalar(mid, alpha, sigma);
        if Fmid < p
            lo = mid;
        else
            hi = mid;
        end
        if hi - lo < 1e-8
            break;
        end
    end
    x = (lo + hi) / 2;
end