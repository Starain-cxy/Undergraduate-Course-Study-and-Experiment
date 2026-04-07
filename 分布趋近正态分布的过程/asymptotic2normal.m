function compare_mean_distribution()
    clear; clc; close all;
    % ==================== 参数设置 ====================
    % 需要绘制的样本量 n
    n_list = [1, 2,3,4,5,6,7,8,12,16,32];
    
    % 离散化步长
    dx = 0.001;
    
    % 定义原始分布 f1（均值为 0，方差为 1）
    % 均匀分布 U[-sqrt(3), sqrt(3)]
    a = -sqrt(3);
    b = sqrt(3);
    f1 = @(x) (x >= a & x <= b) / (b - a);
    
    % 原始支撑区间
    x1 = a : dx : b;
    f1_vals = f1(x1);
    
    % 验证方差
    norm1 = sum(f1_vals) * dx;
    var1  = sum(f1_vals .* x1.^2) * dx;
    fprintf('f1 归一化: %.6f, 方差: %.6f (理论应为 1)\n', norm1, var1);
    
    % 存储结果 (x_grid, f_vals)
    results = cell(length(n_list), 2); 
    
    % 存储 L1 误差
    n_vals = zeros(length(n_list), 1);
    delta_vals = zeros(length(n_list), 1);
    
    % ---------- 迭代卷积 ----------
    curr_x = x1;
    curr_f = f1_vals;
    
    % 保存 n=1
    idx1 = find(n_list == 1);
    if ~isempty(idx1)
        results{idx1, 1} = curr_x;
        results{idx1, 2} = curr_f;
    end
    
    max_n = max(n_list);
    for n = 2 : max_n
        % 卷积计算和的分布
        new_x = (curr_x(1) + x1(1)) : dx : (curr_x(end) + x1(end));
        conv_vals = conv(curr_f, f1_vals) * dx;
        
        % 修正浮点误差导致的长度不匹配
        if length(conv_vals) ~= length(new_x)
            new_x = linspace(curr_x(1)+x1(1), curr_x(end)+x1(end), length(conv_vals));
        end
        
        curr_x = new_x;
        curr_f = conv_vals;
        
        % 数值归一化（防止误差累积）
        curr_f = curr_f / (sum(curr_f) * dx);
        
        % 保存结果
        idx = find(n_list == n);
        if ~isempty(idx)
            results{idx, 1} = curr_x;
            results{idx, 2} = curr_f;
        end
    end
    
    % ---------- 绘图与 L1 误差计算 ----------
    figure('Name', '均值分布与正态对比 (无插值)', 'NumberTitle', 'off');
    
    % 定义用于对比的标准正态分布曲线（高密度点用于绘图）
    x_std = linspace(-4, 4, 1000);
    phi_std = normpdf(x_std, 0, 1);
    
    for i = 1 : length(n_list)
        n = n_list(i);
        x_grid = results{i, 1};   % 原始卷积网格 (和的网格)
        f_vals = results{i, 2};   % 原始概率密度 (和的密度)
        
        % --- 核心变换：从“和”变换为“标准化均值” ---
        % 1. 横坐标变换：Z = S_n / sqrt(n)
        x_transformed = x_grid / sqrt(n);
        
        % 2. 纵坐标变换：f_Z(z) = sqrt(n) * f_S(sqrt(n) * z) 
        y_transformed = f_vals * sqrt(n);
        
        % --- 计算 L1 误差 δ_n = ∫_{-4}^{4} |p_n(x) - φ(x)| dx ---
        % 截取落在 [-4,4] 内的点
        mask = (x_transformed >= -4) & (x_transformed <= 4);
        x_trim = x_transformed(mask);
        y_trim = y_transformed(mask);
        
        % 计算截取点对应的标准正态密度
        phi_trim = normpdf(x_trim, 0, 1);
        
        % 梯形积分求 L1 误差
        delta = trapz(x_trim, abs(y_trim - phi_trim));
        n_vals(i) = n;
        delta_vals(i) = delta;
        
        % --- 绘图 ---
        plot(x_transformed, y_transformed, '.', 'Color', [0.2 0.6 0.9], 'MarkerSize', 5); hold on;
        plot(x_std, phi_std, '--', 'Color', [0.8 0.2 0.2], 'LineWidth', 1.5); hold off;
        
        title(sprintf('n = %d, 均值分布 vs N(0,1)  (δ_n = %.4f)', n, delta));
        xlabel('x'); ylabel('密度');
        legend('数值计算分布', '标准正态', 'Location', 'northeast');
        grid on;
        xlim([-4, 4]);
        
        % 控制翻页
if i < length(n_list)
    disp('按空格键查看下一张图...');
    
    % 使用一个循环来等待空格键
    key = '';
    while ~strcmp(key, ' ')
        % 等待用户交互
        k = waitforbuttonpress;
        
        % 检查是否是键盘按键 (k==1) 且按下的键是空格
        if k == 1
            key = get(gcf, 'CurrentCharacter');
            if ~strcmp(key, ' ')
                disp('请按空格键继续，其他按键无效');
            end
        else
            % 如果是鼠标点击 (k==0)，则忽略，继续等待
            disp('鼠标点击无效，请按空格键继续');
        end
    end
end
    
    % ---------- 绘制 δ_n 随 n 变化的散点图（普通线性坐标） ----------
    figure('Name', 'L1 误差 vs 样本量 n', 'NumberTitle', 'off');
    scatter(n_vals, delta_vals, 50, 'filled', 'MarkerFaceColor', [0.2 0.4 0.8]);
    xlabel('样本量 n');
    ylabel('\delta_n = \int_{-4}^{4} |p_n(x) - \phi(x)| dx');
    title('L1 误差随 n 的变化');
    grid on;
    
    % 添加数值标注（可选）
    for i = 1:length(n_vals)
        text(n_vals(i), delta_vals(i), sprintf('  n=%d', n_vals(i)), ...
            'VerticalAlignment', 'bottom', 'FontSize', 8);
    end
    
    disp('所有图形展示完毕。');
    fprintf('\nL1 误差列表：\n');
    for i = 1:length(n_vals)
        fprintf('n = %2d : δ_n = %.6f\n', n_vals(i), delta_vals(i));
    end
end