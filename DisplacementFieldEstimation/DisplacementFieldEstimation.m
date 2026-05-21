clear; clc; close all;

% DisplacementFieldEstimation.m
% End-to-end pipeline for displacement-field estimation and
% ROI-based visualisation.
%
% Workflow:
%   1) Read ref/t1 ultrasound volumes.
%   2) Estimate 3-D displacement fields (GPU Demons, CPU fallback).
%   3) Render and save two ROI slice views.
%
% Required data layout:
%   ./DisplacementFieldEstimation_data/volumeMatrices/volumeMatrix_ref.mat
%   ./DisplacementFieldEstimation_data/volumeMatrices/volumeMatrix_t1.mat
%   ./DisplacementFieldEstimation_data/DisplacementField/ (output/cache)
%
% Entry:
%   matlab -batch "DisplacementFieldEstimation"

%% === Configuration ===
addpath('func');

% Input/output paths
cfg.dataDir        = './DisplacementFieldEstimation_data';
cfg.volumeDir      = fullfile(cfg.dataDir, 'volumeMatrices');
cfg.flowDir        = fullfile(cfg.dataDir, 'DisplacementField');
cfg.flowref2t1File  = 'D_ref2t1.bin';
cfg.labelMoving     = 'ref';
cfg.labelFixed      = 't1';

% Demons parameters
cfg.scale          = 2;                % preprocessing upsampling factor
cfg.pyrLevels      = 3;                % pyramid depth
cfg.iterations     = [500 400 200];    % per-level iterations (coarse->fine)
cfg.smooth         = 0.8;              % Gaussian regularization sigma
cfg.useGPU         = true;             % true: enable GPU registration first
cfg.resetGPUEachIter = true;           % optional GPU reset between labels

% Physical spacing (mm/voxel):
%   - volume spacing is the native spacing of volumeMatrix
%   - displacement-field spacing follows the registration grid after upsampling
cfg.volumeSpacing  = [0.25 0.25 0.25];
cfg.spacingField   = cfg.volumeSpacing / cfg.scale;

% ROI slice parameters
cfg.roiSlices(1).y = 271;
cfg.roiSlices(1).x = [332, 352];
cfg.roiSlices(1).z = [290, 310];
cfg.roiSlices(2).y = 271;
cfg.roiSlices(2).x = [267, 287];
cfg.roiSlices(2).z = [78, 98];

cfg.roiBorderColors   = [0, 1, 0; 0, 0.5, 1];

cfg.borderLineWidth   = 3;
cfg.titleFontSize     = 24;

cfg.roiUpsampleFactor = 2;
cfg.roiInterpMethod   = 'cubic';

cfg.volumeMatrixClim         = [];
cfg.volumeMatrixColormapName = 'gray';
cfg.volumeMatrixAlpha        = 1.0;
cfg.dColormapName     = 'hot';

cfg.arrowCountTarget  = 300;
cfg.dMagnitudeMin     = 1.0;
cfg.dMagnitudeMax     = 6.0;
cfg.gradientThreshold = 0.0;

cfg.arrowScale            = 0.5;
cfg.arrowLineWidthMin     = 1.5;
cfg.arrowLineWidthMax     = 3.0;
cfg.arrowHeadLenMin       = 0.3;
cfg.arrowHeadLenMax       = 0.5;
cfg.arrowHeadWidMin       = 0.3;
cfg.arrowHeadWidMax       = 0.5;

cfg.preprocessPadZ = 40;
cfg.angleZ         = -30;
cfg.angleX         = 5;
cfg.angleY         = 0;

cfg.figureSize     = [100, 100, 2200, 800];
cfg.roiTitles      = {'t1 upROI', 't1 downROI'};
cfg.outDir         = 'results';
cfg.outName        = 'roi_view.png';


%% === Main pipeline ===
assert(exist(cfg.volumeDir, 'dir') == 7, 'main:noVolumeDir', 'volumeMatrices dir not found: %s', cfg.volumeDir);

if ~exist(cfg.flowDir, 'dir')
    mkdir(cfg.flowDir);
end
if ~exist(cfg.outDir, 'dir')
    mkdir(cfg.outDir);
end

flowref2t1Path  = fullfile(cfg.flowDir, cfg.flowref2t1File);

if exist(flowref2t1Path, 'file') == 2
    fprintf('Found existing flow file, skipping registration: %s\n', flowref2t1Path);
else
    if cfg.useGPU
        [gpuReady, selectedBackend] = try_prepare_gpu(cfg);
        cfg.runtimeBackend = char(selectedBackend);
        if gpuReady
            fprintf('Running registration with resolved backend=%s\n', cfg.runtimeBackend);
        else
            fprintf('GPU backends unavailable. Running CPU fallback\n');
        end
    else
        cfg.runtimeBackend = 'cpu';
        gpuReady = false;
        fprintf('Running registration in CPU-only mode\n');
    end
    register_and_write_label(cfg, cfg.labelMoving, cfg.labelFixed, gpuReady);
end

assert(exist(flowref2t1Path,  'file') == 2, 'main:noFlowAfter',  'Missing flow file: %s', flowref2t1Path);

fprintf('Running visualisation for %s volume\n', cfg.labelFixed);
run_visualisation(cfg);

fprintf('Done. Flow files in %s, figure in %s\n', cfg.flowDir, fullfile(cfg.outDir, cfg.outName));


%% === Registration functions ===

function [ok, backend] = try_prepare_gpu(~)
% try_prepare_gpu  Prepare mexcuda backend once before registration.

ok = false;
backend = "cpu";

if gpuDeviceCount("available") < 1
    fprintf('No available CUDA GPU detected.\n');
    return;
end

g = gpuDevice;
fprintf('Detected GPU: %s, memory %.2f GB\n', g.Name, g.TotalMemory / 1024^3);

try
    if exist('demons3d_cuda', 'file') ~= 3
        fprintf('Compiling CUDA MEX: func/demons3d_cuda.cu\n');
    else
        fprintf('Recompiling CUDA MEX: func/demons3d_cuda.cu\n');
    end
    mexcuda('func/demons3d_cuda.cu', '-output', 'func/demons3d_cuda');
    ok = true;
    backend = "mexcuda";
catch ME
    fprintf('mexcuda unavailable (%s).\n', ME.message);
    fprintf('No usable GPU backend after probing.\n');
end
end


function register_and_write_label(cfg, labelMoving, labelFixed, useGPU)
% register_and_write_label  Compute and persist one label-pair displacement field.
%
% Inputs:
%   cfg         - configuration struct.
%   labelMoving - moving label.
%   labelFixed  - fixed label.
%   useGPU      - whether to attempt GPU registration first.
mov = read_volume_frame(cfg, labelMoving);
fix = read_volume_frame(cfg, labelFixed);

assert(isequal(size(mov), size(fix)), ...
    'main:sizeMismatch', ...
    'Volume size mismatch between %s and %s.', labelMoving, labelFixed);

[u, v, w, usedGPU] = compute_flow(mov, fix, cfg, useGPU);
write_flow_bin(cfg, u, v, w, labelFixed);

if usedGPU && cfg.resetGPUEachIter
    reset(gpuDevice);
end
end


function img = read_volume_frame(cfg, label)
% read_volume_frame  Read one 3-D ultrasound volume by semantic label.
%
% Expected filenames:
%   volumeMatrix_ref.mat and volumeMatrix_t1.mat
% Expected variables inside MAT files:
%   volumeMatrix_ref and volumeMatrix_t1 (or equivalent names in varCandidates)
%
% Output:
%   img  - single-precision 3-D volume.
srcFile = fullfile(cfg.volumeDir, ['volumeMatrix_' label '.mat']);
assert(exist(srcFile, 'file') == 2, 'main:noVolumeFile', 'Volume file not found: %s', srcFile);

varCandidates = {['volumeMatrix' label], label, 'volumeMatrix_t1', 'volumeMatrix_ref'};
S = load(srcFile);

varName = '';
for k = 1:numel(varCandidates)
    if isfield(S, varCandidates{k})
        varName = varCandidates{k};
        break;
    end
end
assert(~isempty(varName), 'main:noDataset', 'No expected volume variable found in %s.', srcFile);

img = single(S.(varName));
assert(ndims(img) == 3, 'main:badVolumeRank', 'Expected 3-D volume in %s.', srcFile);
end


function [u, v, w, usedGPU] = compute_flow(img_mov, img_fixed, cfg, useGPU)
% compute_flow  Estimate dense displacement field from moving to fixed volume.
%
% Steps:
%   1) Spatial upsampling and intensity normalization.
%   2) Histogram matching (moving -> fixed).
%   3) Demons registration (GPU first, custom CPU fallback).
%
% Outputs:
%   u, v, w  - single displacement components.
%   usedGPU  - true when GPU path succeeded.
fixed  = imresize3(img_fixed, cfg.scale);
moving = imresize3(img_mov,   cfg.scale);

fixed  = (fixed  - min(fixed(:)))  / (max(fixed(:))  - min(fixed(:))  + eps);
moving = (moving - min(moving(:))) / (max(moving(:)) - min(moving(:)) + eps);

moving = imhistmatchn(moving, fixed);

if useGPU
    backend = "mexcuda";

    try
        [Dg, ~] = demonsGPU(moving, fixed, cfg.iterations, ...
            'levels', cfg.pyrLevels, ...
            'smoothing', cfg.smooth);
        D = gather(single(Dg));
        usedGPU = true;
    catch ME
        fprintf('GPU backend (%s) failed (%s). Falling back to MATLAB CPU Demons.\n', backend, ME.message);
        D = demonsCPU(moving, fixed, cfg);
        usedGPU = false;
    end
else
    D = demonsCPU(moving, fixed, cfg);
    usedGPU = false;
end

u = single(D(:,:,:,1));
v = single(D(:,:,:,2));
w = single(D(:,:,:,3));
end


function write_flow_bin(cfg, u, v, w, label)
% write_flow_bin  Serialize displacement components to raw single binary.
%
% Output filename:
%   label=cfg.labelFixed -> cfg.flowref2t1File
flow = cat(4, gather(single(u)), gather(single(v)), gather(single(w)));
if strcmpi(label, cfg.labelFixed)
    outFile = fullfile(cfg.flowDir, cfg.flowref2t1File);
else
    error('main:badLabel', 'Unsupported label for write_flow_bin: %s', label);
end

fid = fopen(outFile, 'w');
assert(fid >= 0, 'main:openFailed', 'Failed to open %s for writing.', outFile);
cleanup = onCleanup(@() fclose(fid));

count = fwrite(fid, flow, 'single');
assert(count == numel(flow), ...
    'main:writeFailed', ...
    'Short write for %s: wrote %d of %d elements.', ...
    outFile, count, numel(flow));

fprintf('Wrote flow file: %s\n', outFile);
end


%% === Visualisation functions ===

function run_visualisation(cfg)
% run_visualisation  Render ROI displacement overlays and export the figure.
%
% Input:
%   cfg  - configuration struct from DisplacementFieldEstimation.m.
%
% Steps:
%   1) Load rotated ROI volumeMatrix and displacement slices.
%   2) Build square ROI panel layout in pixel space.
%   3) Render each ROI with displacement arrows.
%   4) Add right-side colorbar panel and export PNG.
roiData = load_draw_roi_data(cfg);

fig = figure('Position', cfg.figureSize, 'Color', 'k', 'InvertHardcopy', 'off');
ax_list = gobjects(1, numel(cfg.roiSlices));

% Figure geometry in pixels.
figW = cfg.figureSize(3);
figH = cfg.figureSize(4);
numRoi = numel(cfg.roiSlices);

leftPx = 80;
rightPx = 40;
topPx = 50;
bottomPx = 50;
gapPx = 40;
sidebarGapPx = 40;
sidebarWpx = 70;

availWpx = figW - leftPx - rightPx - sidebarWpx - sidebarGapPx - gapPx * (numRoi - 1);
availHpx = figH - topPx - bottomPx;
sidePx = min(availWpx / numRoi, availHpx);

totalWpx = numRoi * sidePx + gapPx * (numRoi - 1);
totalBlockWpx = totalWpx + sidebarGapPx + sidebarWpx;
xStartPx = (figW - totalBlockWpx) / 2;
yStartPx = (figH - sidePx) / 2;

% Compute square ROI panel positions and convert to normalized units.
roi_positions = zeros(numRoi, 4);
for i = 1:numRoi
    xPx = xStartPx + (i - 1) * (sidePx + gapPx);
    roi_positions(i, :) = [xPx / figW, yStartPx / figH, sidePx / figW, sidePx / figH];
end

for idx = 1:numel(cfg.roiSlices)
    roi = roiData(idx);
    [vM_roi, Dx, Dz, D_mag, x_coords, z_coords] = upsampleROI(roi.volumeMatrix, roi.Dx, roi.Dz, roi.x, roi.z, cfg.roiUpsampleFactor, cfg.roiInterpMethod);
    volumeMatrix_norm = normalizeData(vM_roi, cfg.volumeMatrixClim);

    ax = axes(fig, 'Position', roi_positions(idx, :), 'Color', 'k');
    ax_list(idx) = ax;
    set(ax, 'PlotBoxAspectRatio', [1 1 1], 'DataAspectRatioMode', 'manual', 'DataAspectRatio', [1 1 1]);
    visualizeDisplacementField( ...
        x_coords, z_coords, volumeMatrix_norm, ...
        Dx, Dz, D_mag, ...
        cfg.volumeMatrixColormapName, cfg.dColormapName, ...
        cfg.arrowScale, ...
        cfg.arrowLineWidthMin, cfg.arrowLineWidthMax, ...
        cfg.arrowHeadLenMin, cfg.arrowHeadLenMax, ...
        cfg.arrowHeadWidMin, cfg.arrowHeadWidMax, ...
        cfg.arrowCountTarget, ...
        cfg.dMagnitudeMin, cfg.dMagnitudeMax, ...
        cfg.gradientThreshold, ...
        cfg.volumeMatrixAlpha, ...
        cfg.roiBorderColors(idx, :), cfg.borderLineWidth);
    title(ax, cfg.roiTitles{idx}, 'Color', 'w', 'FontName', 'Arial', 'FontSize', cfg.titleFontSize);
    set(ax, 'LineWidth', 2, 'XColor', 'none', 'YColor', 'none', 'Box', 'off');
end

% Create a dedicated right-side axis for the displacement magnitude colorbar.
cmapAxX = xStartPx + totalWpx + sidebarGapPx;
cmapAx = axes(fig, 'Position', [cmapAxX / figW, yStartPx / figH, sidebarWpx / figW, sidePx / figH], 'Color', 'k', 'Visible', 'off');
colormap(cmapAx, feval(cfg.dColormapName, 256));
caxis(cmapAx, [0, 1]);
cb = colorbar(cmapAx, 'eastoutside');
cb.Color = 'w';
cb.LineWidth = 1;
cb.FontName = 'Arial';
cb.FontSize = cfg.titleFontSize;
cb.Label.String = 'Norm. disp. amp.';
cb.Label.Color = 'w';
cb.Label.FontName = 'Arial';
cb.Label.FontSize = cfg.titleFontSize;
cb.Ticks = [0, 1];

cbX = (cmapAxX + 38) / figW;
cbY = yStartPx / figH;
cbW = 0.018;
cbH = sidePx / figH;
cb.Position = [cbX, cbY, cbW, cbH];

outPath = fullfile(cfg.outDir, cfg.outName);
exportgraphics(fig, outPath, 'Resolution', 180, 'BackgroundColor', 'black');
fprintf('Saved figure to %s\n', outPath);
end


function [u, v, w] = read_flow_bin_frame(binFile)
% read_flow_bin_frame  Read one displacement-field binary file by memmap.
%
% Input:
%   binFile  - path to raw single-precision flow binary.
%
% Outputs:
%   u, v, w  - displacement components with shape [Nx Ny Nz].
%
% Note:
%   Current implementation expects full-resolution flow size [882 802 722].
assert(exist(binFile, 'file') == 2, 'main:noFlowBin', 'Flow bin file not found: %s', binFile);

fileInfo = dir(binFile);
assert(mod(fileInfo.bytes, 12) == 0, 'main:badFlowBytes', 'Unexpected file size for %s.', binFile);
numel3 = fileInfo.bytes / 4 / 3;

if numel3 == prod([882 802 722])
    flowSize = [882 802 722];
else
    error('main:unknownFlowShape', 'Unsupported flow file size: %s', binFile);
end

Nx = flowSize(1); Ny = flowSize(2); Nz = flowSize(3);
mm = memmapfile(binFile, 'Format', {'single', [Nx, Ny, Nz, 3], 'flow'});

u = mm.Data.flow(:,:,:,1);
v = mm.Data.flow(:,:,:,2);
w = mm.Data.flow(:,:,:,3);
end


function volOut = pad_z_tail(volIn, padCount)
% pad_z_tail  Zero-pad volume on the positive z tail.
%
% Inputs:
%   volIn     - source 3-D volume.
%   padCount  - number of trailing z slices to append.
%
% Output:
%   volOut    - padded volume.
sz = size(volIn);
volOut = zeros(sz(1), sz(2), sz(3) + padCount, 'like', volIn);
volOut(:, :, 1:sz(3)) = volIn;
end


function vol = rotate_preprocessed_volume(vol, cfg)
% rotate_preprocessed_volume  Apply configured 3-axis rotations to a volume.
%
% Input:
%   vol  - source 3-D volume.
%   cfg  - configuration containing angleZ/angleX/angleY.
%
% Output:
%   vol  - rotated volume in loose bounding box with zero fill.
vol = imrotate3(vol, cfg.angleZ, [0 0 1], 'cubic', 'loose', 'FillValues', 0);
vol = imrotate3(vol, cfg.angleX, [0 1 0], 'cubic', 'loose', 'FillValues', 0);
vol = imrotate3(vol, cfg.angleY, [1 0 0], 'cubic', 'loose', 'FillValues', 0);
end


function roiData = load_draw_roi_data(cfg)
% load_draw_roi_data  Build per-ROI volumeMatrix and displacement slices for rendering.
%
% Input:
%   cfg      - configuration with ROI ranges, flow paths, and rotations.
%
% Output:
%   roiData  - struct array with fields x/z/y, volumeMatrix, Dx, Dz, Dmag.
%
% Steps (strictly aligned with reference workflow):
%   1) Read full-resolution D_ref2t1.
%   2) Downsample each displacement component by 0.5.
%   3) Pad z tail by cfg.preprocessPadZ and rotate (Z->X->Y) per component.
%   4) Apply the same pad+rotate flow to volumeMatrix volume.
%   5) Extract ROI slices from rotated fields.
for idx = 1:numel(cfg.roiSlices)
    roiData(idx).x = cfg.roiSlices(idx).x(1):cfg.roiSlices(idx).x(2);
    roiData(idx).z = cfg.roiSlices(idx).z(1):cfg.roiSlices(idx).z(2);
    roiData(idx).y = cfg.roiSlices(idx).y;
    roiData(idx).volumeMatrix = [];
    roiData(idx).Dx = [];
    roiData(idx).Dz = [];
    roiData(idx).Dmag = [];
end

% --- volumeMatrix preprocessing (same geometric flow as displacement) ---
volumeMatrixBase = abs(read_volume_frame(cfg, cfg.labelFixed));
volumeMatrixPad = pad_z_tail(volumeMatrixBase, cfg.preprocessPadZ);
volumeMatrixRot = rotate_preprocessed_volume(volumeMatrixPad, cfg);
for idx = 1:numel(cfg.roiSlices)
    roiData(idx).volumeMatrix = extractRotatedSlice(volumeMatrixRot, cfg.roiSlices(idx));
end
clear volumeMatrixBase volumeMatrixPad volumeMatrixRot;

% --- Displacement preprocessing ---
flowRef2t1 = fullfile(cfg.flowDir, cfg.flowref2t1File);
[uFull, vFull, wFull] = read_flow_bin_frame(flowRef2t1);

uDown = imresize3(uFull, 0.5, 'cubic');
vDown = imresize3(vFull, 0.5, 'cubic');
wDown = imresize3(wFull, 0.5, 'cubic');
clear uFull vFull wFull;

uRot = rotate_preprocessed_volume(pad_z_tail(uDown, cfg.preprocessPadZ), cfg);
vRot = rotate_preprocessed_volume(pad_z_tail(vDown, cfg.preprocessPadZ), cfg);
wRot = rotate_preprocessed_volume(pad_z_tail(wDown, cfg.preprocessPadZ), cfg);
clear uDown vDown wDown;

for idx = 1:numel(cfg.roiSlices)
    roiData(idx).Dx = extractRotatedSlice(uRot, cfg.roiSlices(idx));
    roiData(idx).Dz = extractRotatedSlice(wRot, cfg.roiSlices(idx));
    roiData(idx).Dmag = hypot(roiData(idx).Dx, roiData(idx).Dz);
end
clear uRot vRot wRot;
end


function slice = extractRotatedSlice(vol, roi)
% extractRotatedSlice  Extract one x-z slice at fixed y from rotated volume.
%
% Inputs:
%   vol   - rotated 3-D volume.
%   roi   - struct with fields x=[x0 x1], z=[z0 z1], y.
%
% Output:
%   slice - 2-D ROI slice indexed as vol(x, y, z).
x = roi.x(1):roi.x(2);
z = roi.z(1):roi.z(2);
y = roi.y;
slice = squeeze(vol(x, y, z));
end


function [vM_up, Dx_up, Dz_up, Dmag_up, x_up, z_up] = upsampleROI(vM, Dx, Dz, x, z, up_factor, method)
% upsampleROI  Interpolate ROI textures/vectors onto a denser x-z grid.
%
% Inputs:
%   vM, Dx, Dz  - ROI volumeMatrix and displacement components.
%   x, z        - original coordinate vectors.
%   up_factor   - upsampling factor per axis.
%   method      - interpolation method (default 'cubic').
%
% Outputs:
%   vM_up, Dx_up, Dz_up - upsampled ROI arrays.
%   Dmag_up             - magnitude map of upsampled displacement.
%   x_up, z_up          - dense coordinate vectors.
if nargin < 7
    method = 'cubic';
end

[X, Z] = meshgrid(x, z);

x_up = linspace(min(x), max(x), numel(x) * up_factor);
z_up = linspace(min(z), max(z), numel(z) * up_factor);
[Xq, Zq] = meshgrid(x_up, z_up);

vM_up  = interp2(X, Z, vM',  Xq, Zq, method)';
Dx_up = interp2(X, Z, Dx', Xq, Zq, method)' * up_factor;
Dz_up = interp2(X, Z, Dz', Xq, Zq, method)' * up_factor;

Dmag_up = hypot(Dx_up, Dz_up);
end


function visualizeDisplacementField(x, z, volumeMatrix, Dx, Dz, Dmag, volumeMatrix_cmap, arrow_cmap, scale, lw_min, lw_max, hlen_min, hlen_max, hwid_min, hwid_max, N_target, dmin, dmax, grad_th, ~, border_color, border_line_width)
% visualizeDisplacementField  Draw volumeMatrix ROI with displacement arrows.
%
% Inputs:
%   x, z, volumeMatrix    - ROI coordinate axes and normalized volumeMatrix image.
%   Dx, Dz, Dmag         - displacement components and magnitude.
%   volumeMatrix_cmap     - colormap name for volumeMatrix underlay.
%   arrow_cmap           - colormap name for displacement arrows.
%   scale                - arrow length scaling.
%   lw_*, hlen_*, hwid_* - arrow style ranges mapped by displacement magnitude.
%   N_target             - target number of sampled arrows.
%   dmin, dmax           - displacement magnitude display range.
%   grad_th              - gradient threshold for arrow candidate filtering.
%   border_color/width   - ROI border style.
%
% Notes:
%   - Candidate arrows are ranked by volumeMatrix gradient magnitude.
%   - All arrow lines are clipped to the ROI box.
[gx, gz] = gradient(double(volumeMatrix));
gmag = normalizeData(hypot(gx, gz), []);

valid = (Dmag >= dmin) & (Dmag <= dmax) & (gmag > grad_th);
idx = find(valid);
[~, order] = sort(gmag(idx), 'descend');
idx = idx(order(1:min(N_target, numel(idx))));

imagesc(x, z, volumeMatrix', [0 1]);
axis image; axis xy;
colormap(gca, feval(volumeMatrix_cmap, 256));
axis off;
set(gca, 'XColor', 'none', 'YColor', 'none');
set(gca, 'Color', 'k');
hold on;

x_min = min(x);
x_max = max(x);
z_min = min(z);
z_max = max(z);

cmap = feval(arrow_cmap, 256);
[X, Z] = meshgrid(x, z);

all_indices = idx;
[nx, nz] = size(Dx);

for k = 1:numel(all_indices)
    curr_idx = all_indices(k);
    [i, j] = ind2sub([nx, nz], curr_idx);

    dx = Dx(i,j) * scale;
    dz = Dz(i,j) * scale;
    mag = hypot(dx, dz);
    if mag < eps, continue; end

    ux = dx / mag;
    uz = dz / mag;
    px = -uz;
    pz = ux;

    w = (Dmag(i,j) - dmin) / (dmax - dmin);
    w = max(0, min(1, w));
    lw   = lw_min   + w * (lw_max   - lw_min);
    hlen = hlen_min + w * (hlen_max - hlen_min);
    hwid = hwid_min + w * (hwid_max - hwid_min);
    col = cmap(round(w * 255) + 1, :);

    x0 = X(j,i);
    z0 = Z(j,i);
    x1_raw = x0 + dx;
    z1_raw = z0 + dz;
    [x1, z1] = clipLineToBox(x0, z0, x1_raw, z1_raw, x_min, x_max, z_min, z_max);
    if isempty(x1)
        continue;
    end
    lw_draw = max(lw, 0.01);
    line([x0, x1], [z0, z1], 'Color', col, 'LineWidth', lw_draw);

    bx = x1 - hlen * ux;
    bz = z1 - hlen * uz;

    left_x = bx + (hwid / 2) * px;
    left_z = bz + (hwid / 2) * pz;
    right_x = bx - (hwid / 2) * px;
    right_z = bz - (hwid / 2) * pz;

    [left_x, left_z] = clipLineToBox(x1, z1, left_x, left_z, x_min, x_max, z_min, z_max);
    [right_x, right_z] = clipLineToBox(x1, z1, right_x, right_z, x_min, x_max, z_min, z_max);

    if ~isempty(left_x)
        line([x1, left_x], [z1, left_z], 'Color', col, 'LineWidth', lw_draw);
    end
    if ~isempty(right_x)
        line([x1, right_x], [z1, right_z], 'Color', col, 'LineWidth', lw_draw);
    end
end

pixel_dx = localPixelStep(x);
pixel_dz = localPixelStep(z);
x_edge = [x_min - pixel_dx / 2, x_max + pixel_dx / 2];
z_edge = [z_min - pixel_dz / 2, z_max + pixel_dz / 2];

line([x_edge(1), x_edge(2)], [z_edge(1), z_edge(1)], 'Color', border_color, 'LineWidth', border_line_width, 'Clipping', 'off');
line([x_edge(1), x_edge(2)], [z_edge(2), z_edge(2)], 'Color', border_color, 'LineWidth', border_line_width, 'Clipping', 'off');
line([x_edge(1), x_edge(1)], [z_edge(1), z_edge(2)], 'Color', border_color, 'LineWidth', border_line_width, 'Clipping', 'off');
line([x_edge(2), x_edge(2)], [z_edge(1), z_edge(2)], 'Color', border_color, 'LineWidth', border_line_width, 'Clipping', 'off');

hold off;
end


function [x1_clip, z1_clip] = clipLineToBox(x0, z0, x1, z1, x_min, x_max, z_min, z_max)
% clipLineToBox  Clip a line segment endpoint against an axis-aligned box.
%
% Inputs:
%   (x0,z0)       - segment start point.
%   (x1,z1)       - segment end point.
%   x_min/x_max   - x bounds of clipping box.
%   z_min/z_max   - z bounds of clipping box.
%
% Outputs:
%   x1_clip,z1_clip - clipped endpoint; empty when fully outside.
dx = x1 - x0;
dz = z1 - z0;
t0 = 0;
t1 = 1;

[t0, t1, ok] = clipTest(-dx, x0 - x_min, t0, t1);
if ~ok, x1_clip = []; z1_clip = []; return; end
[t0, t1, ok] = clipTest(dx, x_max - x0, t0, t1);
if ~ok, x1_clip = []; z1_clip = []; return; end
[t0, t1, ok] = clipTest(-dz, z0 - z_min, t0, t1);
if ~ok, x1_clip = []; z1_clip = []; return; end
[t0, t1, ok] = clipTest(dz, z_max - z0, t0, t1);
if ~ok, x1_clip = []; z1_clip = []; return; end

x1_clip = x0 + t1 * dx;
z1_clip = z0 + t1 * dz;
end


function [t0, t1, ok] = clipTest(p, q, t0, t1)
% clipTest  Liang-Barsky clipping helper for one inequality.
%
% Inputs:
%   p, q    - line/edge relation terms.
%   t0, t1  - current parametric interval on segment.
%
% Outputs:
%   t0, t1  - updated interval.
%   ok      - false when interval becomes empty.
if p == 0
    ok = q >= 0;
    return;
end

r = q / p;
if p < 0
    if r > t1
        ok = false;
        return;
    end
    if r > t0
        t0 = r;
    end
else
    if r < t0
        ok = false;
        return;
    end
    if r < t1
        t1 = r;
    end
end
ok = true;
end


function out = normalizeData(in, clim)
% normalizeData  Min-max normalize to [0,1] with optional fixed limits.
%
% Inputs:
%   in    - numeric array.
%   clim  - [] for data-driven limits, or [min max].
%
% Output:
%   out   - normalized and clamped array in [0,1].
if isempty(clim)
    mn = min(in(:));
    mx = max(in(:));
else
    mn = clim(1);
    mx = clim(2);
end
out = (in - mn) / max(mx - mn, eps);
out = max(0, min(1, out));
end


function step = localPixelStep(coords)
% localPixelStep  Estimate pixel pitch from coordinate vector.
%
% Input:
%   coords  - coordinate samples along one axis.
%
% Output:
%   step    - median spacing, or 1 for degenerate vectors.
if numel(coords) < 2
    step = 1;
else
    step = median(diff(coords));
end
end
