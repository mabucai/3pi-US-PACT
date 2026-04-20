%% Dynamic Functional Connectivity (dFC) Analysis 
clear;close all force;clc;
addpath('utils');

%% 1. Data Loading & States Loading
rng(7); 
load('dFC_data/dFC_data.mat');
% Phase Extraction
signal_total = dfc_phase(PA_signal); 

% States Loading
[N, ~, K] = size(state_mats); % N=14, K=4
num_frames = size(signal_total, 3); % 6000

%% 2. Extracting features 
fprintf('>>> Extracting features from 6000 frames...\n');

mask = tril(true(N), -1); 
num_features = sum(mask(:));
new_features = zeros(num_frames, num_features); 
old_centroids = zeros(K, num_features);

for t = 1:num_frames
    temp_mat = signal_total(:, :, t);
    new_features(t, :) = temp_mat(mask);
end

% Feature extraction for the 4 identified centroids
for k = 1:K
    temp_center = state_mats(:, :, k);
    old_centroids(k, :) = temp_center(mask);
end

%% 3. Classification (Manhattan distance)
fprintf('>>> Classifying 6000 frames based on Manhattan distance...\n');

distances = pdist2(new_features, old_centroids, 'cityblock');
[~, new_cluster_idx] = min(distances, [], 2);

fprintf('>>> Classification complete.\n');

%% 4. Statistical results (Occupancy)
new_occ = zeros(1, K);
for k = 1:K
    new_occ(k) = mean(new_cluster_idx == k) * 100;
    fprintf('State %d Occupancy: %.2f%%\n', k, new_occ(k));
end

%% 5. Visualization: Connectivity & Occupancy

fprintf('>>> Generating Final Visualization (Connectivity & Overall Occupancy)...\n');
% --- A. ---
figure('Name', 'State Centroids', 'Position', [100, 100, 1200, 300], 'Color', 'w');
myCmap = morandi_blue_red(256, 5, 1.05); 
labels_short = {'ITGa (L)','OL (L)','ITGp (L)', 'CGa (L)', 'CGp (L)', 'FL (L)', 'PL (L)', ...
                'PL (R)', 'FL (R)', 'CGp (R)', 'CGa (R)', 'ITGp (R)', 'OL (R)','ITGa (R)'};

for k = 1:4
    subplot(1, 4, k);
    imagesc(state_mats(:, :, k)); 
    colormap(myCmap); 
    caxis([-1 1]);
    axis image; 
    set(gca, 'XTick', [], 'YTick', [], 'XColor', 'none', 'YColor', 'none'); 
    for i = 1:N
        text(0.3, i, labels_short{i}, 'FontSize', 8, 'HorizontalAlignment', 'right');
        text(i, 14.7, labels_short{i}, 'FontSize', 8, 'Rotation', 90, 'HorizontalAlignment', 'right');
    end
    title(['State ', num2str(k)], 'FontSize', 12);
end

h = colorbar('Position', [0.93, 0.2, 0.015, 0.6]); 
h.Label.String = 'cos( \Deltaphase)';
h.Label.FontSize = 10;
sgtitle('Recurrent dynamic connectivity states', 'FontSize', 14, 'FontWeight', 'bold');


hFig = state_plot(new_cluster_idx);

dfc_matrix_show(signal_total);



%% 6. Helper Function
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