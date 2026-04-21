function D = demonsCPU(moving, fixed, cfg)
% demonsCPU
% CPU implementation of 3-D additive Demons registration (single precision).
%
% This function is the CPU fallback path used by main.m when the CUDA path
% is unavailable (no GPU / MEX build failure / GPU out-of-memory).
%
% Inputs:
%   moving  - moving volume (before), expected 3-D single
%   fixed   - fixed volume  (after),  expected 3-D single
%   cfg     - configuration struct from main.m:
%             cfg.iterations  : per-level iteration count (e.g., [500 400 200])
%             cfg.pyrLevels   : number of pyramid levels
%             cfg.smooth      : Gaussian smoothing sigma for displacement field
%
% Output:
%   D       - dense displacement field, size [Nx Ny Nz 3], single
%             D(:,:,:,1) = u (x/column displacement)
%             D(:,:,:,2) = v (y/row displacement)
%             D(:,:,:,3) = w (z/slice displacement)
%
% Algorithmic correspondence with GPU version:
%   - Same core Demons update form
%   - Same multiresolution coarse-to-fine strategy
%   - Same validity thresholds used in CUDA kernel:
%       |diff| >= 1e-3, denominator >= 1e-9
%   - Same iterative Gaussian regularisation of displacement components
%
% Main implementation differences vs. CUDA path:
%   - Warping/interpolation is done by MATLAB interp3 on CPU
%   - Smoothing is done by imgaussfilt3 on CPU
%   - No custom fused kernels / no GPU memory layout optimisations

% -------------------------------------------------------------------------
% 1) Parse and normalise iteration schedule
% -------------------------------------------------------------------------
if numel(cfg.iterations) ~= cfg.pyrLevels
    iters = repmat(cfg.iterations(1), 1, cfg.pyrLevels);
else
    iters = cfg.iterations;
end

levels = cfg.pyrLevels;

% -------------------------------------------------------------------------
% 2) Build Gaussian pyramid (coarse -> fine)
% -------------------------------------------------------------------------
movPyr = cell(1, levels);
fixPyr = cell(1, levels);

for l = 1:levels
    % At level l, scale factor is 0.5^(levels-l):
    % l=1 coarsest, l=levels finest.
    s = 0.5^(levels - l);
    if s == 1
        movPyr{l} = single(moving);
        fixPyr{l} = single(fixed);
    else
        movPyr{l} = single(imresize3(moving, s, 'linear'));
        fixPyr{l} = single(imresize3(fixed,  s, 'linear'));
    end
end

% -------------------------------------------------------------------------
% 3) Initialise accumulated displacement at coarsest level
% -------------------------------------------------------------------------
u = zeros(size(movPyr{1}), 'single'); % x / column displacement
v = zeros(size(movPyr{1}), 'single'); % y / row displacement
w = zeros(size(movPyr{1}), 'single'); % z / slice displacement

% -------------------------------------------------------------------------
% 4) Coarse-to-fine optimisation
% -------------------------------------------------------------------------
for l = 1:levels
    movL = movPyr{l};
    fixL = fixPyr{l};

    % Upsample displacement from previous (coarser) level.
    % Geometric scaling and magnitude scaling both need factor 2.
    if l > 1
        targetSize = size(movL);
        u = 2 * single(imresize3(u, targetSize, 'linear'));
        v = 2 * single(imresize3(v, targetSize, 'linear'));
        w = 2 * single(imresize3(w, targetSize, 'linear'));
    end

    % Gradient of fixed image (kept constant within one pyramid level).
    % MATLAB gradient output order for 3-D array is [row, col, slice].
    [dFix_dRow, dFix_dCol, dFix_dSlice] = gradient(fixL);
    gx = dFix_dCol;
    gy = dFix_dRow;
    gz = dFix_dSlice;

    nIter = iters(l);

    % ---------------------------------------------------------------------
    % 5) Demons iterations at current level
    % ---------------------------------------------------------------------
    for it = 1:nIter
        % 5.1 Warp moving image with current displacement field.
        movWarp = warp_volume_linear(movL, u, v, w);


        % 5.2 Compute additive Demons update.
        % diff = F - M_warped
        diff = fixL - movWarp;

        % denominator = |grad(F)|^2 + diff^2
        denom = gx.^2 + gy.^2 + gz.^2 + diff.^2;

        % Same stability/flat-region gating as CUDA version.
        valid = abs(diff) >= 1e-3 & denom >= 1e-9 & isfinite(movWarp);

        du = zeros(size(u), 'single');
        dv = zeros(size(v), 'single');
        dw = zeros(size(w), 'single');

        factor = zeros(size(u), 'single');
        factor(valid) = diff(valid) ./ denom(valid);

        du(valid) = factor(valid) .* gx(valid);
        dv(valid) = factor(valid) .* gy(valid);
        dw(valid) = factor(valid) .* gz(valid);

        % 5.3 Accumulate field.
        u = u + du;
        v = v + dv;
        w = w + dw;

        % 5.4 Gaussian regularisation (separable filtering internally).
        if cfg.smooth > 0
            u = imgaussfilt3(u, cfg.smooth, 'FilterDomain', 'spatial', 'Padding', 'replicate');
            v = imgaussfilt3(v, cfg.smooth, 'FilterDomain', 'spatial', 'Padding', 'replicate');
            w = imgaussfilt3(w, cfg.smooth, 'FilterDomain', 'spatial', 'Padding', 'replicate');
        end
    end
end

% -------------------------------------------------------------------------
% 6) Pack displacement components into [Nx Ny Nz 3]
% -------------------------------------------------------------------------
D = cat(4, single(u), single(v), single(w));

end


function out = warp_volume_linear(vol, u, v, w)
% warp_volume_linear
% Trilinear warping on CPU with out-of-bound value set to NaN.
%
% vol : source 3-D volume
% u/v/w: displacement components aligned with vol grid

[rows, cols, slices] = size(vol);
[cg, rg, sg] = meshgrid(1:cols, 1:rows, 1:slices);

xq = cg + u;
yq = rg + v;
zq = sg + w;

out = interp3(vol, xq, yq, zq, 'linear', NaN);
out = single(out);
end
