function fig = state_plot(data_states)
% STATE_PLOT Plots state transition trajectories as blocks
% Input:
%   data_states - 1D state index vector (e.g., 1xframe or framex1)

    % 1. Initialization
    COMMON_LEN = length(data_states);
    npg_colors = [
        0.55, 0.40, 0.75; ... % State 1
        0.10, 0.65, 0.55; ... % State 2
        0.90, 0.35, 0.25; ... % State 3
        0.30, 0.75, 0.85      % State 4
    ];

    % 2. Figure and Axes Setup
    fig = figure('Position', [100, 100, 1000, 300], 'Color', 'w');
    ax = axes('Position', [0.1, 0.2, 0.85, 0.7], 'Color', 'w');
    hold on;

    % 3. Plotting Logic
    % --- 3.1 State Transition Blocks ---
    BAR_HALF_HEIGHT = 0.5; 
    
    change_points = find(diff(data_states) ~= 0).'; 
    changes = [1, change_points + 1, COMMON_LEN + 1];
    
    for k = 1:length(changes)-1
        idx_s = changes(k);
        idx_e = changes(k+1) - 1;
        curr_s = data_states(idx_s);
        
        if curr_s > 0 && curr_s <= size(npg_colors, 1)
            patch([idx_s idx_e idx_e idx_s], ...
                  [curr_s-BAR_HALF_HEIGHT curr_s-BAR_HALF_HEIGHT curr_s+BAR_HALF_HEIGHT curr_s+BAR_HALF_HEIGHT], ...
                  npg_colors(curr_s, :), 'EdgeColor', 'none', 'FaceAlpha', 1.0);
        end
    end

    % --- 3.2 Horizontal Dividers ---
    line_y = 1.5:1:3.5;
    for y = line_y
        line([1, COMMON_LEN], [y, y], 'Color', 'k', 'LineWidth', 1.0);
    end

    % --- 3.3 Styling, Borders and Labels ---
    xlim([1 COMMON_LEN]);
    ylim([0.5 4.5]);
    
    line([1, COMMON_LEN, COMMON_LEN, 1, 1], [0.5, 0.5, 4.5, 4.5, 0.5], 'Color', 'k', 'LineWidth', 1.2);

    DATA_MINUTES = 10; 
    tick_pos = linspace(1, COMMON_LEN, DATA_MINUTES + 1);
    tick_labels = 0:DATA_MINUTES; %
    
    xlabel('Time (min)', 'FontSize', 12); 

    set(ax, ...
        'YDir', 'reverse', ...                   
        'YTick', 1:4, ...                       
        'YTickLabel', {'State 1', 'State 2', 'State 3', 'State 4'}, ... 
        'XTick', tick_pos, ...               
        'XTickLabel', tick_labels, ...           
        'TickDir', 'out', ...                    
        'XColor', 'k', 'YColor', 'k', ...
        'LineWidth', 1.2, ...
        'Box', 'off');
        
    ax.LooseInset = [0 0 0 0];
    sgtitle('Dynamic transitions of connectivity states', 'FontSize', 14, 'FontWeight', 'bold');
    fprintf('Plot Complete: Labels and Time axis updated.\n');
end