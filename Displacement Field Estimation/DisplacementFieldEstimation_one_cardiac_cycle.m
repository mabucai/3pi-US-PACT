clear; clc;

% DisplacementFieldEstimation_cycle_compute.m
% Pure displacement-field computation for a 4-D cardiac cycle volume.
% No visualisation is generated.

%% === Configuration ===
addpath('func');

cfg.dataDir        = './DisplacementFieldEstimation_one_cardiac_cycle_data';
cfg.volumeSubDir   = 'volumeMatrices';
cfg.flowDir        = fullfile(cfg.dataDir, 'DisplacementField');
cfg.volumeFileName = 'volumeMatrices_one_cardiac_cycle.mat';
cfg.volumeVarName  = 'volumeMatrices_one_cardiac_cycle';

cfg.scale          = 2;
cfg.pyrLevels      = 3;
cfg.iterations     = [500 400 200];
cfg.smooth         = 0.8;
cfg.useGPU         = true;
cfg.resetGPUEachIter = true;

%% === Path resolve ===
cfg.volumeDir = fullfile(cfg.dataDir, cfg.volumeSubDir);
if exist(cfg.volumeDir, 'dir') ~= 7
    error('cycle:noVolumeDir', 'volumeMatrices dir not found in %s', cfg.dataDir);
end

if ~exist(cfg.flowDir, 'dir')
    mkdir(cfg.flowDir);
end

srcFile = fullfile(cfg.volumeDir, cfg.volumeFileName);
assert(exist(srcFile, 'file') == 2, 'cycle:noVolumeFile', 'Volume file not found: %s', srcFile);

%% === Dataset info (HDF5) ===
[datasetPath, datasetSize] = resolve_h5_dataset(srcFile, cfg.volumeVarName);
assert(numel(datasetSize) == 4, 'cycle:badRank', 'Expected 4-D dataset, got %d-D.', numel(datasetSize));

Nx = datasetSize(1);
Ny = datasetSize(2);
Nz = datasetSize(3);
Nt = datasetSize(4);

assert(Nt >= 2, 'cycle:badFrames', 'Need at least 2 frames, got %d.', Nt);

if cfg.useGPU
    [gpuReady, selectedBackend] = try_prepare_gpu();
    cfg.runtimeBackend = char(selectedBackend);
    if gpuReady
        fprintf('Running pairwise registration with backend=%s\n', cfg.runtimeBackend);
    else
        fprintf('GPU unavailable. Running CPU fallback\n');
    end
else
    cfg.runtimeBackend = 'cpu';
    gpuReady = false;
    fprintf('Running in CPU-only mode\n');
end

%% === Pairwise compute: (1->2), (3->4), ... ===
for t = 1:2:(Nt - 1)
    chunk = h5read(srcFile, datasetPath, [1 1 1 t], [Nx Ny Nz 2]);
    moving = single(chunk(:,:,:,1));
    fixed  = single(chunk(:,:,:,2));

    [u, v, w, usedGPU] = compute_flow_pair(moving, fixed, cfg, gpuReady);

    outFile = fullfile(cfg.flowDir, sprintf('D_%02dto%02d.bin', t, t + 1));
    write_flow_bin(outFile, u, v, w);
    fprintf('Wrote %s\n', outFile);

    if usedGPU && cfg.resetGPUEachIter
        reset(gpuDevice);
    end
end

fprintf('Done.\n');


function [ok, backend] = try_prepare_gpu()
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


function [u, v, w, usedGPU] = compute_flow_pair(img_mov, img_fixed, cfg, useGPU)
fixed  = imresize3(img_fixed, cfg.scale);
moving = imresize3(img_mov,   cfg.scale);

fixed  = (fixed  - min(fixed(:)))  / (max(fixed(:))  - min(fixed(:))  + eps);
moving = (moving - min(moving(:))) / (max(moving(:)) - min(moving(:)) + eps);

moving = imhistmatchn(moving, fixed);

if useGPU
    try
        Dg = demonsGPU(moving, fixed, cfg.iterations, ...
            'levels', cfg.pyrLevels, ...
            'smoothing', cfg.smooth);
        D = gather(single(Dg));
        usedGPU = true;
    catch ME
        fprintf('GPU backend (%s) failed (%s). Falling back to MATLAB CPU Demons.\n', string(cfg.runtimeBackend), ME.message);
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


function write_flow_bin(outFile, u, v, w)
u = gather(single(u));
v = gather(single(v));
w = gather(single(w));
fid = fopen(outFile, 'w');
assert(fid >= 0, 'cycle:openFailed', 'Failed to open %s for writing.', outFile);
cleanup = onCleanup(@() fclose(fid));

count = 0;
count = count + fwrite(fid, u, 'single');
count = count + fwrite(fid, v, 'single');
count = count + fwrite(fid, w, 'single');
expected = numel(u) + numel(v) + numel(w);
assert(count == expected, 'cycle:writeFailed', ...
    'Short write for %s: wrote %d of %d elements.', outFile, count, expected);
end


function [datasetPath, datasetSize] = resolve_h5_dataset(h5file, datasetName)
info = h5info(h5file);
[datasetPath, datasetSize] = walk_group(info, datasetName, '/');
assert(~isempty(datasetPath), 'cycle:noDataset', 'Dataset %s not found in %s', datasetName, h5file);
end


function [foundPath, foundSize] = walk_group(groupInfo, datasetName, prefix)
foundPath = '';
foundSize = [];

for i = 1:numel(groupInfo.Datasets)
    ds = groupInfo.Datasets(i);
    if strcmp(ds.Name, datasetName)
        if strcmp(prefix, '/')
            foundPath = ['/' ds.Name];
        else
            foundPath = [prefix '/' ds.Name];
        end
        foundSize = ds.Dataspace.Size;
        return;
    end
end

for i = 1:numel(groupInfo.Groups)
    g = groupInfo.Groups(i);
    [foundPath, foundSize] = walk_group(g, datasetName, g.Name);
    if ~isempty(foundPath)
        return;
    end
end
end
