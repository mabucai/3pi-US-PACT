function D = demonsCPU(moving, fixed, cfg)
backend = "native";
if isfield(cfg, 'cpuBackend') && ~isempty(cfg.cpuBackend)
    backend = string(cfg.cpuBackend);
end

switch lower(char(backend))
    case 'native'
        opts.iters = resolveIterations(cfg);
        opts.levels = resolveLevels(cfg);
        opts.smooth = resolveSmooth(cfg);
        D = demonsCPU_native(single(moving), single(fixed), opts);
    case 'legacy'
        D = demonsCPU_legacy(single(moving), single(fixed), cfg);
    otherwise
        error('demonsCPU:badBackend', 'Unsupported CPU backend: %s', backend);
end
end


function D = demonsCPU_native(moving, fixed, opts)
checkPyramidSize(moving, opts.levels);

if opts.levels > 1
    [fixed, padVec] = padToPyramid(fixed, opts.levels);
    moving = padToPyramid(moving, opts.levels);

    coarseFactor = 0.5^(opts.levels - 1);
    coarseSz = round(size(fixed) * coarseFactor);
    u = zeros(coarseSz, 'single');
    v = zeros(coarseSz, 'single');
    w = zeros(coarseSz, 'single');

    for level = 1:opts.levels
        movL = pyramidDownsample(moving, level, opts.levels);
        fixL = pyramidDownsample(fixed, level, opts.levels);

        if level > 1
            u = scaleField(u, 2);
            v = scaleField(v, 2);
            w = scaleField(w, 2);
        end

        [u, v, w] = runLevelDemons(movL, fixL, u, v, w, opts.iters(level), opts.smooth, level, opts.levels);
    end

    u = cropPad(u, padVec);
    v = cropPad(v, padVec);
    w = cropPad(w, padVec);
else
    sz = size(fixed);
    u = zeros(sz, 'single');
    v = zeros(sz, 'single');
    w = zeros(sz, 'single');
    [u, v, w] = runLevelDemons(moving, fixed, u, v, w, opts.iters, opts.smooth, 1, 1);
end

D = cat(4, u, v, w);
end


function [u, v, w] = runLevelDemons(moving, fixed, u, v, w, nIter, sigma, levelIdx, numLevels)
[gx, gy, gz] = fixedGradient3(fixed);
[rows, cols, slices] = size(fixed);
[baseX, baseY, baseZ] = meshgrid(single(1:cols), single(1:rows), single(1:slices));
probeIters = min(5, nIter);
levelTimer = tic;
estimatePrinted = false;

for iter = 1:nIter
    movingWarp = warpVolumeLinear(moving, baseX, baseY, baseZ, u, v, w);
    diff = fixed - movingWarp;
    denom = gx.^2 + gy.^2 + gz.^2 + diff.^2;
    valid = abs(diff) >= single(1e-3) & denom >= single(1e-9) & isfinite(movingWarp);

    factor = zeros(size(diff), 'single');
    factor(valid) = diff(valid) ./ denom(valid);

    u(valid) = u(valid) + factor(valid) .* gx(valid);
    v(valid) = v(valid) + factor(valid) .* gy(valid);
    w(valid) = w(valid) + factor(valid) .* gz(valid);

    if sigma > 0
        u = gaussSep3(u, sigma);
        v = gaussSep3(v, sigma);
        w = gaussSep3(w, sigma);
    end

    if ~estimatePrinted && iter == probeIters
        elapsed = toc(levelTimer);
        estimatedTotal = elapsed / probeIters * nIter;
        fprintf('demonsCPU level %d/%d estimated total time: %.2f s (based on first %d iterations)\n', ...
            levelIdx, numLevels, estimatedTotal, probeIters);
        estimatePrinted = true;
    end
end
end


function warped = warpVolumeLinear(vol, baseX, baseY, baseZ, u, v, w)
xq = baseX + u;
yq = baseY + v;
zq = baseZ + w;
warped = interp3(vol, xq, yq, zq, 'linear', NaN);
warped = single(warped);
end


function out = gaussSep3(in, sigma)
radius = ceil(3 * double(sigma));
if radius <= 0
    out = in;
    return
end
kernel = gaussianKernel1d(single(sigma), radius);
out = convReplicateDim(in, kernel, 2);
out = convReplicateDim(out, kernel, 1);
out = convReplicateDim(out, kernel, 3);
out = single(out);
end


function kernel = gaussianKernel1d(sigma, radius)
x = single(-radius:radius);
kernel = exp(-(x .* x) ./ (single(2) * sigma * sigma));
kernel = kernel ./ sum(kernel);
kernel = reshape(single(kernel), 1, []);
end


function out = convReplicateDim(in, kernel, dim)
radius = floor((numel(kernel) - 1) / 2);
padSize = [0 0 0];
padSize(dim) = radius;
padded = padarray(in, padSize, 'replicate', 'both');
shape = [1 1 1];
shape(dim) = numel(kernel);
flt = reshape(kernel, shape);
out = convn(padded, flt, 'valid');
out = single(out);
end


function [gx, gy, gz] = fixedGradient3(img)
img = single(img);
[rows, cols, slices] = size(img);

% gy: rows / dim1
gy = zeros(size(img), 'single');
gy(1,:,:) = img(2,:,:) - img(1,:,:);
gy(rows,:,:) = img(rows,:,:) - img(rows-1,:,:);
if rows > 2
    gy(2:rows-1,:,:) = (img(3:rows,:,:) - img(1:rows-2,:,:)) * 0.5;
end

% gx: cols / dim2
gx = zeros(size(img), 'single');
gx(:,1,:) = img(:,2,:) - img(:,1,:);
gx(:,cols,:) = img(:,cols,:) - img(:,cols-1,:);
if cols > 2
    gx(:,2:cols-1,:) = (img(:,3:cols,:) - img(:,1:cols-2,:)) * 0.5;
end

% gz: slices / dim3
gz = zeros(size(img), 'single');
gz(:,:,1) = img(:,:,2) - img(:,:,1);
gz(:,:,slices) = img(:,:,slices) - img(:,:,slices-1);
if slices > 2
    gz(:,:,2:slices-1) = (img(:,:,3:slices) - img(:,:,1:slices-2)) * 0.5;
end
end


function B = pyramidDownsample(A, level, numLevels)
B = resizeAA(A, 0.5 .^ (numLevels - level));
end


function Dout = scaleField(Din, factor)
Dout = resizeAA(Din, factor) .* single(factor);
end


function out = resizeAA(in, factor)
classIn = class(in);
if factor == 1
    out = in;
    return
elseif factor < 1
    I = fftshift(fftn(in));
    H = butterworth(0.5 * factor, 2, size(in));
    in = ifftn(ifftshift(I .* H), 'symmetric');
end

if ~strcmp(classIn, 'single')
    in = single(in);
end

factor = single(factor);
srcSz = size(in);
dstSz = round(double(srcSz) * double(factor));
Rout = imref3d(dstSz);

T = single([factor 0 0 0; 0 factor 0 0; 0 0 factor 0; 0 0 0 1]);
Tx = single(Rout.XIntrinsicLimits(1) * double(factor) - Rout.XIntrinsicLimits(1));
Ty = single(Rout.YIntrinsicLimits(1) * double(factor) - Rout.YIntrinsicLimits(1));
Tz = single(Rout.ZIntrinsicLimits(1) * double(factor) - Rout.ZIntrinsicLimits(1));

tI2W = single([1 0 0 0; 0 1 0 0; 0 0 1 0; Tx Ty Tz 1]);
tComp = single(tI2W / T);
tComp(:,4) = single([0; 0; 0; 1]);

xq = single(1:Rout.ImageSize(2));
yq = single(1:Rout.ImageSize(1));
zq = single(1:Rout.ImageSize(3));
[gx, gy, gz] = meshgrid(xq, yq, zq);
uvw = zeros([numel(gx), 4], 'single');
uvw(:,1) = gx(:);
uvw(:,2) = gy(:);
uvw(:,3) = gz(:);
uvw(:,4) = single(1);
xyz = uvw * tComp;
sx = xyz(:,1);
sy = xyz(:,2);
sz_ = xyz(:,3);

pad = 3;
in = padarray(in, [pad pad pad]);
sx = sx + pad;
sy = sy + pad;
sz_ = sz_ + pad;

out = interp3(in, sx, sy, sz_, 'linear');
out = reshape(out, dstSz);
out = cast(out, classIn);
out = single(out);
end


function H = butterworth(Do, n, outsize)
nr = outsize(1);
nc = outsize(2);
np = outsize(3);
u = single(normFreq(nc));
v = single(normFreq(nr));
w = single(normFreq(np));
[U, V, W] = meshgrid(u, v, w);
D = sqrt(U.^2 + V.^2 + W.^2);
Do = single(Do);
n = single(n);
H = single(1) ./ (single(1) + (D ./ Do).^(single(2) * n));
end


function u = normFreq(N)
if mod(N, 2)
    u = linspace(-0.5 + 1/(2*N), 0.5 - 1/(2*N), N);
else
    u = linspace(-0.5, 0.5 - 1/N, N);
end
u = single(u);
end


function checkPyramidSize(vol, numLevels)
minSz = 2^numLevels;
if any(size(vol) < minSz)
    error('demonsCPU:imageTooSmall', ...
        'Each spatial dimension must be at least %d voxels for %d pyramid level(s).', ...
        minSz, numLevels);
end
end


function [out, padVec] = padToPyramid(vol, numLevels)
factor = 2^numLevels;
sz = size(vol);
newSz = ceil(sz / factor) * factor;
padVec = newSz - sz;
out = padarray(vol, padVec, 'replicate', 'post');
end


function out = cropPad(in, padVec)
out = in(1:end-padVec(1), 1:end-padVec(2), 1:end-padVec(3));
end


function iters = resolveIterations(cfg)
if isfield(cfg, 'iterations') && ~isempty(cfg.iterations)
    iters = double(cfg.iterations);
else
    iters = 100;
end

levels = resolveLevels(cfg);
if isscalar(iters)
    iters = repmat(iters, 1, levels);
end
if numel(iters) ~= levels
    error('demonsCPU:itersMismatch', 'iterations must be a scalar or have length pyrLevels (%d).', levels);
end
end


function levels = resolveLevels(cfg)
if isfield(cfg, 'pyrLevels') && ~isempty(cfg.pyrLevels)
    levels = double(cfg.pyrLevels);
else
    levels = 1;
end
end


function smooth = resolveSmooth(cfg)
if isfield(cfg, 'smooth') && ~isempty(cfg.smooth)
    smooth = double(cfg.smooth);
else
    smooth = 1.0;
end
end


function D = demonsCPU_legacy(moving, fixed, cfg)
if numel(cfg.iterations) ~= cfg.pyrLevels
    iters = repmat(cfg.iterations(1), 1, cfg.pyrLevels);
else
    iters = cfg.iterations;
end

levels = cfg.pyrLevels;
movPyr = cell(1, levels);
fixPyr = cell(1, levels);

for l = 1:levels
    s = 0.5^(levels - l);
    if s == 1
        movPyr{l} = single(moving);
        fixPyr{l} = single(fixed);
    else
        movPyr{l} = single(imresize3(moving, s, 'linear'));
        fixPyr{l} = single(imresize3(fixed,  s, 'linear'));
    end
end

u = zeros(size(movPyr{1}), 'single');
v = zeros(size(movPyr{1}), 'single');
w = zeros(size(movPyr{1}), 'single');

for l = 1:levels
    movL = movPyr{l};
    fixL = fixPyr{l};

    if l > 1
        targetSize = size(movL);
        u = 2 * single(imresize3(u, targetSize, 'linear'));
        v = 2 * single(imresize3(v, targetSize, 'linear'));
        w = 2 * single(imresize3(w, targetSize, 'linear'));
    end

    [dFix_dRow, dFix_dCol, dFix_dSlice] = gradient(fixL);
    gx = dFix_dCol;
    gy = dFix_dRow;
    gz = dFix_dSlice;

    nIter = iters(l);
    for it = 1:nIter
        movWarp = warp_volume_linear_legacy(movL, u, v, w);
        diff = fixL - movWarp;
        denom = gx.^2 + gy.^2 + gz.^2 + diff.^2;
        valid = abs(diff) >= 1e-3 & denom >= 1e-9 & isfinite(movWarp);

        du = zeros(size(u), 'single');
        dv = zeros(size(v), 'single');
        dw = zeros(size(w), 'single');
        factor = zeros(size(u), 'single');
        factor(valid) = diff(valid) ./ denom(valid);
        du(valid) = factor(valid) .* gx(valid);
        dv(valid) = factor(valid) .* gy(valid);
        dw(valid) = factor(valid) .* gz(valid);

        u = u + du;
        v = v + dv;
        w = w + dw;

        if cfg.smooth > 0
            u = imgaussfilt3(u, cfg.smooth, 'FilterDomain', 'spatial', 'Padding', 'replicate');
            v = imgaussfilt3(v, cfg.smooth, 'FilterDomain', 'spatial', 'Padding', 'replicate');
            w = imgaussfilt3(w, cfg.smooth, 'FilterDomain', 'spatial', 'Padding', 'replicate');
        end
    end
end

D = cat(4, single(u), single(v), single(w));
end


function out = warp_volume_linear_legacy(vol, u, v, w)
[rows, cols, slices] = size(vol);
[cg, rg, sg] = meshgrid(1:cols, 1:rows, 1:slices);
xq = cg + u;
yq = rg + v;
zq = sg + w;
out = interp3(vol, xq, yq, zq, 'linear', NaN);
out = single(out);
end
