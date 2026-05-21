clear; clc;

% DisplacementFieldEstimation_one_cardiac_cycle.m
% Pure displacement-field computation for a cardiac cycle volume sequence.
% Each frame is stored as an individual 3-D MAT file under volumeMatrices/.
% No visualisation is generated.

%% === Configuration ===
addpath('func');

cfg.dataDir          = './DisplacementFieldEstimation_one_cardiac_cycle_data';
cfg.volumeSubDir     = 'volumeMatrices';
cfg.flowDir          = fullfile(cfg.dataDir, 'DisplacementField');
cfg.filePattern      = 'volumeMatrix_*.mat';
cfg.volumeVarName    = 'volumeMatrix';
cfg.expectedNumFrame = 60;

cfg.scale            = 2;
cfg.pyrLevels        = 3;
cfg.iterations       = [500 400 200];
cfg.smooth           = 0.8;
cfg.useGPU           = true;
cfg.resetGPUEachIter = true;

%% === Path resolve ===
cfg.volumeDir = fullfile(cfg.dataDir, cfg.volumeSubDir);
if exist(cfg.volumeDir, 'dir') ~= 7
    error('cycle:noVolumeDir', 'volumeMatrices dir not found in %s', cfg.dataDir);
end

if ~exist(cfg.flowDir, 'dir')
    mkdir(cfg.flowDir);
end

frameFiles = dir(fullfile(cfg.volumeDir, cfg.filePattern));
assert(~isempty(frameFiles), ...
    'cycle:noVolumeFrames', ...
    'No frame MAT files found under %s with pattern %s.', ...
    cfg.volumeDir, cfg.filePattern);

[frameNumbers, order] = sort(extract_frame_numbers({frameFiles.name}));
frameFiles = frameFiles(order);

assert(all(diff(frameNumbers) == 1), ...
    'cycle:nonContinuousFrames', ...
    'Expected consecutive frame files, got: %s', mat2str(frameNumbers));
assert(frameNumbers(1) == 1, ...
    'cycle:badFrameStart', ...
    'Expected frame numbering to start from 1, got %d.', frameNumbers(1));

Nt = numel(frameFiles);
assert(Nt >= 2, 'cycle:badFrames', 'Need at least 2 frames, got %d.', Nt);

if ~isempty(cfg.expectedNumFrame)
    assert(Nt == cfg.expectedNumFrame, ...
        'cycle:badFrameCount', ...
        'Expected %d frames, found %d.', cfg.expectedNumFrame, Nt);
end

sampleVolume = read_volume_frame_file(fullfile(cfg.volumeDir, frameFiles(1).name), cfg);
[Nx, Ny, Nz] = size(sampleVolume);

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
for idx = 1:2:(Nt - 1)
    tMoving = frameNumbers(idx);
    tFixed  = frameNumbers(idx + 1);

    moving = read_volume_frame_file(fullfile(cfg.volumeDir, frameFiles(idx).name), cfg);
    fixed  = read_volume_frame_file(fullfile(cfg.volumeDir, frameFiles(idx + 1).name), cfg);

    assert(isequal(size(moving), [Nx Ny Nz]), ...
        'cycle:sizeMismatch', ...
        'Unexpected size in %s.', frameFiles(idx).name);
    assert(isequal(size(fixed), [Nx Ny Nz]), ...
        'cycle:sizeMismatch', ...
        'Unexpected size in %s.', frameFiles(idx + 1).name);

    [u, v, w, usedGPU] = compute_flow_pair(moving, fixed, cfg, gpuReady);

    outFile = fullfile(cfg.flowDir, sprintf('D_%02dto%02d.bin', tMoving, tFixed));
    write_flow_bin(outFile, u, v, w);
    fprintf('Wrote %s\n', outFile);

    if usedGPU && cfg.resetGPUEachIter
        reset(gpuDevice);
    end
end

fprintf('Done.\n');


function frameNumbers = extract_frame_numbers(fileNames)
frameNumbers = zeros(1, numel(fileNames));

for idx = 1:numel(fileNames)
    tokens = regexp(fileNames{idx}, '^volumeMatrix_(\d+)\.mat$', 'tokens', 'once');
    assert(~isempty(tokens), ...
        'cycle:badFrameFileName', ...
        'Unexpected frame file name: %s', fileNames{idx});
    frameNumbers(idx) = str2double(tokens{1});
end
end


function img = read_volume_frame_file(srcFile, cfg)
assert(exist(srcFile, 'file') == 2, 'cycle:noVolumeFile', 'Volume file not found: %s', srcFile);

S = load(srcFile);
varCandidates = {cfg.volumeVarName, 'volumeMatrix', 'volume', 'img'};

varName = '';
for k = 1:numel(varCandidates)
    if isfield(S, varCandidates{k})
        varName = varCandidates{k};
        break;
    end
end

if isempty(varName)
    fields = fieldnames(S);
    if numel(fields) == 1
        varName = fields{1};
    end
end

assert(~isempty(varName), ...
    'cycle:noDataset', ...
    'No expected volume variable found in %s.', srcFile);

img = single(S.(varName));
assert(ndims(img) == 3, ...
    'cycle:badVolumeRank', ...
    'Expected 3-D volume in %s.', srcFile);
end


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
