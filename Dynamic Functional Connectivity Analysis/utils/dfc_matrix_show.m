function dfc_matrix_show(data)
% DFC_MATRIX_SHOW Plays dynamic phase matrix animation
% Input:
%   data     - 3D matrix [14 x 14 x Frames]

    % 2. Layout Initialization
    fig = figure('Color','k', 'Position', [100, 100, 1050, 850], 'Visible', 'on'); 
    
    % Custom colormap
    myCmap = morandi_blue_red(256, 5, 1.05); 
    colormap(myCmap); 
    
    ax = gca;
    set(ax, 'Position', [0.12, 0.15, 0.75, 0.75]); 
    ax.Color = 'k'; 
    ax.YDir = 'reverse'; 
    hold on;

    % --- Initial Heatmap ---
    hImg = imagesc(data(:,:,1));
    clim([-1 1]);       
    xlim([0.5, 14.5]);
    ylim([0.5, 14.5]);
    axis square; 

    % --- Grid Lines ---
    nLabels = 14;  
    gridColor = [0.3 0.3 0.3]; 
    for i = 0:nLabels
        line([i+0.5, i+0.5], [0.5, nLabels+0.5], 'Color', gridColor, 'LineWidth', 1.2);
        line([0.5, nLabels+0.5], [i+0.5, i+0.5], 'Color', gridColor, 'LineWidth', 1.2);
    end

    % --- Labels ---
    labels_short = {'ITGa (L)','OL (L)','ITGp (L)', 'CGa (L)', 'CGp (L)', 'FL (L)', 'PL (L)', ...
                    'PL (R)', 'FL (R)', 'CGp (R)', 'CGa (R)', 'ITGp (R)', 'OL (R)','ITGa (R)'};
    
    fontSize = 22; 
    set(ax, 'XTick', [], 'YTick', [], 'XColor', 'none', 'YColor', 'none');
    for i = 1:nLabels
        % Left labels
        text(0.3, i, labels_short{i}, 'Color', 'w', 'FontSize', fontSize, 'HorizontalAlignment', 'right');
        % Bottom labels (Rotated)
        text(i, 14.7, labels_short{i}, 'Color', 'w', 'FontSize', fontSize,'Rotation', 90, 'HorizontalAlignment', 'right');
    end

    % --- Colorbar ---
    hBar = colorbar;
    set(hBar, 'Ticks', [-1, 1]); 
    set(hBar, 'Color', 'w', 'FontSize', fontSize, 'Visible', 'on'); 
    hBar.Label.String = 'cos( \Deltaphase)';
    hBar.Label.Color = 'w';
    hBar.Label.FontSize =22;
    hBar.Label.Rotation = 90;
    hBar.Label.Position(1) = 2.0;
    % --- Titles and Texts ---
    title('Dynamic phase matrix', 'Color', 'w', 'FontSize', 24);
    hTime = text(12.5, 0, sprintf('T = %.1f s', 0.1), ...
                 'Color', 'w', 'FontSize', 20, 'HorizontalAlignment', 'left');
    num_time_points = size(data, 3);
    for t = 1:num_time_points
        % Update heatmap data
        set(hImg, 'CData', data(:,:,t));
        
        % Update time text only
        current_time = t * 0.1;
        set(hTime, 'String', sprintf('T = %.1f s', current_time));
        
        drawnow limitrate;
        pause(0.004);
    end
end



function cmap = morandi_blue_red(n, saturation_factor, light_factor)
    base_colors = [70,100,160; 120,150,190; 170,190,210; 210,210,210; 220,200,190; 200,160,150; 170,120,110]/255 * light_factor;
    for i = 1:size(base_colors, 1)
        hsv = rgb2hsv(base_colors(i, :));
        hsv(2) = min(hsv(2) * saturation_factor, 0.8);
        base_colors(i, :) = hsv2rgb(hsv);
    end
    cmap = zeros(n, 3);
    node_positions = linspace(1, n, size(base_colors, 1));
    for c = 1:3
        cmap(:, c) = interp1(node_positions, base_colors(:, c), 1:n, 'pchip');
    end
    cmap = min(max(cmap, 0), 1);
end