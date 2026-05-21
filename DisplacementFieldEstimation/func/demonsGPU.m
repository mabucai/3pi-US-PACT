function [D, movingReg] = demonsGPU(moving, fixed, varargin)
% demonsGPU  GPU-accelerated 3-D Demons non-rigid image registration.
%
%   This function was written with reference to MATLAB's imregdemons (Image
%   Processing Toolbox) for the multi-resolution pyramid structure and
%   coordinate conventions. All calls to MathWorks internal APIs have been
%   replaced with independent implementations. The core Demons iteration is
%   handled by a custom CUDA MEX kernel (demons3d_cuda).
%   Note: parameter names and calling conventions intentionally differ from
%   imregdemons.
%
%   D = demonsGPU(MOVING, FIXED, N) registers the gpuArray volume MOVING to
%   FIXED using N Demons iterations and returns the 4-D displacement field D
%   of size [rows x cols x slices x 3], where D(:,:,:,1/2/3) are the x/y/z
%   components respectively.
%
%   [D, MOVINGREG] = demonsGPU(...) also returns the warped moving volume.
%
%   demonsGPU(..., Name, Value) accepts the following optional parameters:
%     'levels'    – number of multi-resolution pyramid levels (default 1)
%     'smoothing' – Gaussian sigma for displacement field regularisation (default 1.0)
%
%   Both MOVING and FIXED must be gpuArray volumes of equal size and a
%   supported numeric class (uint8, uint16, uint32, int8, int16, int32,
%   single, double, or logical). Intermediate computation is performed in
%   single precision.
%
%   Compile the CUDA MEX accelerator before first use:
%     mexcuda func/demons3d_cuda.cu -output func/demons3d_cuda
%
%   References:
%     [1] J.-P. Thirion, "Image matching as a diffusion process: an analogy
%         with Maxwell's demons", Medical Image Analysis, 2(3), 1998.
%     [2] T. Vercauteren et al., "Diffeomorphic Demons: Efficient
%         Non-parametric Image Registration", NeuroImage, 45(S1), 2009.
%     [3] MathWorks, "imregdemons", Image Processing Toolbox documentation,
%         https://www.mathworks.com/help/images/ref/imregdemons.html

narginchk(2, inf);

[moving, fixed] = checkInputs(moving, fixed);

% Parse positional N and optional name-value parameters.
opts = parseOptions(varargin{:});

% Validate that the volume is large enough for the requested pyramid depth.
checkPyramidSize(moving, opts.levels);

classMoving = underlyingType(moving);

% Promote to single for intermediate computation.
fixed  = single(fixed);
moving = single(moving);

[D, movingReg] = pyramidDemons(moving, fixed, opts, classMoving);

end


% --------------------------------------------------------------------------

function [D, movingReg] = pyramidDemons(moving, fixed, opts, classMoving)
% pyramidDemons  Run Demons registration at multiple resolutions.
% Builds a Gaussian image pyramid, estimates the displacement field at the
% coarsest level, then progressively upsamples and refines toward full resolution.

if opts.levels > 1

    [fixed, padVec] = padToPyramid(fixed,  opts.levels);
    moving          = padToPyramid(moving, opts.levels);

    % Initialise directly at the coarsest pyramid level to avoid unnecessary
    % resize/filter work on an all-zero field.
    coarseFactor = 0.5^(opts.levels - 1);
    coarseSz     = round(size(fixed) * coarseFactor);
    Da_x         = gpuArray.zeros(coarseSz, 'single');
    Da_y         = gpuArray.zeros(coarseSz, 'single');
    Da_z         = gpuArray.zeros(coarseSz, 'single');

    for p = 1:opts.levels

        % Downsample images to the current pyramid level.
        mov_p = pyramidDownsample(moving, p, opts.levels);
        fix_p = pyramidDownsample(fixed,  p, opts.levels);

        if p > 1
            % Upsample the displacement field before refining at the next
            % (finer) resolution level; scaleField also rescales magnitudes.
            Da_x = scaleField(Da_x, 2);
            Da_y = scaleField(Da_y, 2);
            Da_z = scaleField(Da_z, 2);
        end

        % Run Demons iterations at the current resolution level.
        [Da_x, Da_y, Da_z] = demons3d_cuda(mov_p, fix_p, opts.iters(p), ...
            opts.smoothing, Da_x, Da_y, Da_z);

        wait(gpuDevice);
        clear mov_p fix_p
    end

    % Remove padding voxels added to make dimensions divisible by 2^levels.
    Da_x   = cropPad(Da_x,   padVec);
    Da_y   = cropPad(Da_y,   padVec);
    Da_z   = cropPad(Da_z,   padVec);
    moving = cropPad(moving, padVec);

else
    % Single-resolution: initialise field with zeros and run directly.
    sz   = size(fixed);
    Da_x = gpuArray.zeros(sz, 'single');
    Da_y = gpuArray.zeros(sz, 'single');
    Da_z = gpuArray.zeros(sz, 'single');
    [Da_x, Da_y, Da_z] = demons3d_cuda(moving, fixed, opts.iters, ...
        opts.smoothing, Da_x, Da_y, Da_z);
end

% Stack x/y/z components into a single 4-D field [rows x cols x slices x 3].
D = cat(4, Da_x, Da_y, Da_z);

if nargout > 1
    movingReg = warpMoving(moving, Da_x, Da_y, Da_z);
    % Cast the warped image back to the original input class.
    movingReg = cast(movingReg, classMoving);
end

end


% --------------------------------------------------------------------------

function out = resizeAA(in, factor, varargin)
% resizeAA  Resize a 3-D volume with anti-aliasing.
% Applies a 2nd-order Butterworth low-pass filter before downsampling
% (cutoff = 0.5 * factor) to suppress aliasing, then resamples via an
% affine warp using linear interpolation. For upsampling (factor > 1) the
% filter step is skipped.

classIn = underlyingType(in);

if factor == 1
    out = in;
    return
elseif factor < 1
    % Move to frequency domain and apply low-pass filter before downsampling.
    I  = fftshift(fftn(in));
    H  = butterworth(0.5 * factor, 2, size(in));
    in = ifftn(ifftshift(I .* H), 'symmetric');
end

% Promote to single for interpolation and keep intermediates in single.
if ~strcmp(classIn, 'single')
    in = single(in);
end

% Build the affine scale transform that maps output intrinsic coordinates
% to input intrinsic coordinates (replicates imwarp behaviour).
factor = single(factor);
srcSz = size(in);
dstSz = round(double(srcSz) * double(factor));
Rout  = imref3d(dstSz);

T  = single([factor 0 0 0; 0 factor 0 0; 0 0 factor 0; 0 0 0 1]);
Tx = single(Rout.XIntrinsicLimits(1) * double(factor) - Rout.XIntrinsicLimits(1));
Ty = single(Rout.YIntrinsicLimits(1) * double(factor) - Rout.YIntrinsicLimits(1));
Tz = single(Rout.ZIntrinsicLimits(1) * double(factor) - Rout.ZIntrinsicLimits(1));

tI2W  = single([1 0 0 0; 0 1 0 0; 0 0 1 0; Tx Ty Tz 1]);
tComp = single(tI2W / T);
tComp(:, 4) = single([0; 0; 0; 1]);

% Build output coordinate grid on the GPU.
xq = single(gpuArray.colon(1, Rout.ImageSize(2)));
yq = single(gpuArray.colon(1, Rout.ImageSize(1)));
zq = single(gpuArray.colon(1, Rout.ImageSize(3)));
[gx, gy, gz] = meshgrid(xq, yq, zq);

uvw = gpuArray.zeros([numel(gx), 4], 'single');
uvw(:, 1) = gx(:);
uvw(:, 2) = gy(:);
uvw(:, 3) = gz(:);
uvw(:, 4) = single(1);
clear gx gy gz

xyz  = uvw * tComp;
clear uvw
sx   = xyz(:, 1);
sy   = xyz(:, 2);
sz_  = xyz(:, 3);
clear xyz

% Pad by 3 voxels on each side to support linear interpolation at borders.
pad = 3;
in  = padarray(in, [pad pad pad]);
sx  = sx  + pad;
sy  = sy  + pad;
sz_ = sz_ + pad;

out = interp3(in, sx, sy, sz_, 'linear');
out = reshape(out, dstSz);
out = cast(out, classIn);

end


% --------------------------------------------------------------------------

function warped = warpMoving(moving, Da_x, Da_y, Da_z)
% warpMoving  Warp MOVING by the displacement field (Da_x, Da_y, Da_z).
% Builds sampling coordinates from the accumulated field and applies trilinear
% interpolation (padded by 1 voxel, boundary value = 0).

sz = size(Da_x);
xq = single(gpuArray.colon(1, sz(2)));
yq = single(gpuArray.colon(1, sz(1)));
zq = single(gpuArray.colon(1, sz(3)));
[gx, gy, gz] = meshgrid(xq, yq, zq);

% Add displacement to intrinsic coordinates. The +1 offset accounts for the
% 1-voxel padding added by padarray below (shifts grid into padded space).
Ux = gx + Da_x + single(1);
Uy = gy + Da_y + single(1);
Uz = gz + Da_z + single(1);

warped = interp3(padarray(moving, [1 1 1]), Ux, Uy, Uz, 'linear', single(0));

end


% --------------------------------------------------------------------------

function out = cropPad(in, padVec)
% cropPad  Remove pyramid padding from a volume or field component.
out = in(1:end - padVec(1), 1:end - padVec(2), 1:end - padVec(3));
end


% --------------------------------------------------------------------------

function B = pyramidDownsample(A, level, numLevels)
% pyramidDownsample  Downsample A to the given pyramid level.
% level == numLevels yields the coarsest (smallest) image.
B = resizeAA(A, 0.5 .^ (numLevels - level));
end


% --------------------------------------------------------------------------

function Dout = scaleField(Din, factor)
% scaleField  Spatially rescale a displacement field component.
% After geometric resampling, displacement magnitudes are scaled by the same
% factor to remain consistent with the new voxel spacing.
Dout = resizeAA(Din, factor) .* factor;
end


% --------------------------------------------------------------------------

function H = butterworth(Do, n, outsize)
% butterworth  3-D Butterworth low-pass filter in the frequency domain.
%   H = butterworth(Do, n, outsize) returns a real-valued filter of size
%   outsize with normalised cutoff frequency Do (0 < Do <= 0.5) and order n.
nr = outsize(1);  nc = outsize(2);  np = outsize(3);

u = single(normFreq(nc));
v = single(normFreq(nr));
w = single(normFreq(np));
[U, V, W] = meshgrid(u, v, w);

D = sqrt(U.^2 + V.^2 + W.^2);
Do = single(Do);
n = single(n);
H = single(1) ./ (single(1) + (D ./ Do).^(single(2) * n));
end


% --------------------------------------------------------------------------

function u = normFreq(N)
% normFreq  Return N normalised frequency samples in [-0.5, 0.5).
if mod(N, 2)
    u = linspace(-0.5 + 1/(2*N), 0.5 - 1/(2*N), N);
else
    u = linspace(-0.5, 0.5 - 1/N, N);
end
end


% --------------------------------------------------------------------------

function [moving, fixed] = checkInputs(moving, fixed)
% checkInputs  Validate and move both input volumes to the GPU.
% Enforces matching dimensionality (2-D or 3-D), supported numeric class,
% and real/finite/non-empty attributes.
validClasses = {'uint8','uint16','uint32','int8','int16','int32','single','double','logical'};
validAttribs = {'real','finite','nonempty'};

moving = gpuArray(moving);
fixed  = gpuArray(fixed);

validateattributes(moving, validClasses, validAttribs, mfilename, 'MOVING', 1);
validateattributes(fixed,  validClasses, validAttribs, mfilename, 'FIXED',  2);

badDim = @(im) ndims(im) < 2 || ndims(im) > 3;
if badDim(moving) || badDim(fixed)
    error('demonsGPU:invalidDimensions', 'MOVING and FIXED must be 2-D or 3-D volumes.');
end
if ~isequal(ndims(moving), ndims(fixed))
    error('demonsGPU:dimensionMismatch', 'MOVING and FIXED must have the same number of dimensions.');
end
end


% --------------------------------------------------------------------------

function opts = parseOptions(varargin)
% parseOptions  Parse demonsGPU optional parameters.
% First positional argument (optional): iterations count/vector.
% Supported name-value pairs with defaults:
%   'levels'    – pyramid depth  (default 1)
%   'smoothing' – Gaussian sigma (default 1.0)

p = inputParser();
p.addOptional('iters',    100,   @(x) isnumeric(x) && all(x >= 0));
p.addParameter('levels',    1,     @(x) isnumeric(x) && isscalar(x) && x >= 1);
p.addParameter('smoothing', 1.0,   @(x) isnumeric(x) && isscalar(x) && x >= 0);
p.parse(varargin{:});

opts         = p.Results;
opts.levels  = double(opts.levels);

% Expand scalar iters to a vector matching the pyramid depth.
if isscalar(opts.iters)
    opts.iters = repmat(opts.iters, 1, opts.levels);
end
if numel(opts.iters) ~= opts.levels
    error('demonsGPU:itersMismatch', ...
        'iters must be a scalar or a vector of length levels (%d).', opts.levels);
end
end


% --------------------------------------------------------------------------

function checkPyramidSize(vol, numLevels)
% checkPyramidSize  Check that the volume is large enough for the pyramid.
% Each level halves all spatial dimensions, so each must be >= 2^numLevels.
minSz = 2^numLevels;
if any(size(vol) < minSz)
    error('demonsGPU:imageTooSmall', ...
        'Each spatial dimension must be at least %d voxels for %d pyramid level(s).', ...
        minSz, numLevels);
end
end


% --------------------------------------------------------------------------

function [out, padVec] = padToPyramid(vol, numLevels)
% padToPyramid  Pad volume so each dimension is divisible by 2^numLevels.
% padVec = [padR, padC, padS] records how many voxels were added per
% dimension so cropPad can restore the original size afterwards.
factor = 2^numLevels;
sz     = size(vol);
newSz  = ceil(sz / factor) * factor;
padVec = newSz - sz;
out    = padarray(vol, padVec, 'replicate', 'post');
end


