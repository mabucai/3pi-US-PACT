/*
 * demons3d_cuda.cu
 * CUDA MEX implementation of the 3D Demons image registration algorithm.
 *
 * Inputs  (called from demonsGPU.m via MEX):
 *   prhs[0]  moving   – gpuArray (single), the moving volume  [rows x cols x slices]
 *   prhs[1]  fixed    – gpuArray (single), the fixed/target volume [rows x cols x slices]
 *   prhs[2]  N        – scalar double, number of Demons iterations
 *   prhs[3]  sigma    – scalar double, Gaussian smoothing std-dev for the accumulated field
 *                        (cast to float inside the MEX entry point)
 *   prhs[4]  Dax      – gpuArray (single), initial x-displacement field (can be zeros)
 *   prhs[5]  Day      – gpuArray (single), initial y-displacement field (can be zeros)
 *   prhs[6]  Daz      – gpuArray (single), initial z-displacement field (can be zeros)
 *
 * Outputs:
 *   plhs[0]  Dax_out  – updated x-displacement field (gpuArray)
 *   plhs[1]  Day_out  – updated y-displacement field (gpuArray)
 *   plhs[2]  Daz_out  – updated z-displacement field (gpuArray)
 *
 * Algorithm per iteration:
 *   1. Warp moving image with current field (trilinear interpolation).
 *   2. Compute additive Demons update from intensity difference and fixed gradient.
 *   3. Smooth the accumulated field with a separable Gaussian (replicate border).
 *
 * Features:
 *   - Fixed-image gradient computed once before the loop (not per-iteration).
 *   - Fused warp + update kernel reduces global memory round-trips.
 *   - Separable 1-D Gaussian convolution (3 passes) for field regularisation.
 *   - Column-major memory layout matches MATLAB gpuArray convention.
 *
 * Compile with:
 *   mexcuda func/demons3d_cuda.cu -output func/demons3d_cuda
 */

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <algorithm>

#define BLOCK_SIZE_X 8
#define BLOCK_SIZE_Y 8
#define BLOCK_SIZE_Z 8

// Error checking macro
#define cudaCheckError() { \
    cudaError_t e = cudaGetLastError(); \
    if(e != cudaSuccess) { \
        mexErrMsgIdAndTxt("demons3d_cuda:cudaError", "CUDA Error: %s", cudaGetErrorString(e)); \
    } \
}

// --------------------------------------------------------------------------
// Device Functions
// --------------------------------------------------------------------------

// Trilinear interpolation on a column-major 3-D volume.
// Coordinates (r, c, s) are 0-based floating-point indices.
// Returns NaN when any of the eight surrounding lattice points lies outside
// the volume, matching MATLAB's interp3(..., 'linear', NaN) behaviour.
__device__ float sample_vol_trilinear(const float* vol, int rows, int cols, int slices, float r, float c, float s) {
    int r0 = (int)floor(r);
    int c0 = floor(c);
    int s0 = floor(s);
    
    int r1 = r0 + 1;
    int c1 = c0 + 1;
    int s1 = s0 + 1;

    // All eight neighbours must be inside the volume; otherwise return NaN.
    if (r0 < 0 || c0 < 0 || s0 < 0 || r1 >= rows || c1 >= cols || s1 >= slices) {
        return nanf(""); // Return NaN for extrapolation
    }

    float dr = r - r0;
    float dc = c - c0;
    float ds = s - s0;

    // Helper to get value (Column-Major layout: r + c*rows + s*rows*cols)
    auto get = [&](int y, int x, int z) -> float {
        return vol[(long long)z * rows * cols + (long long)x * rows + y];
    };

    float v000 = get(r0, c0, s0);
    float v100 = get(r1, c0, s0);
    float v010 = get(r0, c1, s0);
    float v110 = get(r1, c1, s0);
    float v001 = get(r0, c0, s1);
    float v101 = get(r1, c0, s1);
    float v011 = get(r0, c1, s1);
    float v111 = get(r1, c1, s1);

    // Interpolate along Rows (y)
    float c00 = v000 * (1.0f - dr) + v100 * dr;
    float c10 = v010 * (1.0f - dr) + v110 * dr;
    float c01 = v001 * (1.0f - dr) + v101 * dr;
    float c11 = v011 * (1.0f - dr) + v111 * dr;

    // Interpolate along Cols (x)
    float c0_ = c00 * (1.0f - dc) + c10 * dc;
    float c1_ = c01 * (1.0f - dc) + c11 * dc;

    // Interpolate along Slices (z)
    return c0_ * (1.0f - ds) + c1_ * ds;
}

// --------------------------------------------------------------------------
// Kernels
// --------------------------------------------------------------------------

// 1. Fixed-image gradient kernel (central differences, forward/backward at borders).
// Follows MATLAB's gradient() convention:
//   Gy (rows / dim1): (F(r+1) - F(r-1)) / 2
//   Gx (cols / dim2): (F(c+1) - F(c-1)) / 2
//   Gz (slices/ dim3): (F(s+1) - F(s-1)) / 2
__global__ void kernel_gradient(const float* img, float* gx, float* gy, float* gz,
                                int rows, int cols, int slices) {
    int c = blockIdx.x * blockDim.x + threadIdx.x; // Cols (x)
    int r = blockIdx.y * blockDim.y + threadIdx.y; // Rows (y)
    int s = blockIdx.z * blockDim.z + threadIdx.z; // Slices (z)

    if (r >= rows || c >= cols || s >= slices) return;

    long long idx = (long long)s * rows * cols + (long long)c * rows + r;

    // Gradient Y (Rows)
    if (r == 0) gy[idx] = img[idx+1] - img[idx];
    else if (r == rows - 1) gy[idx] = img[idx] - img[idx-1];
    else gy[idx] = (img[idx+1] - img[idx-1]) * 0.5f;

    // Gradient X (Cols)
    long long stride_col = rows;
    if (c == 0) gx[idx] = img[idx + stride_col] - img[idx];
    else if (c == cols - 1) gx[idx] = img[idx] - img[idx - stride_col];
    else gx[idx] = (img[idx + stride_col] - img[idx - stride_col]) * 0.5f;

    // Gradient Z (Slices)
    long long stride_slice = (long long)rows * cols;
    if (s == 0) gz[idx] = img[idx + stride_slice] - img[idx];
    else if (s == slices - 1) gz[idx] = img[idx] - img[idx - stride_slice];
    else gz[idx] = (img[idx + stride_slice] - img[idx - stride_slice]) * 0.5f;
}

// 2. Fused warp-and-update kernel.
// For each voxel:
//   a) Warp the moving image to the current accumulated displacement field.
//   b) Compute the additive Demons update step using the Thirion / diffeomorphic
//      Demons force formula:  du = (F - M_warped) * grad(F) / (|grad(F)|^2 + (F - M_warped)^2)
//   c) Add the update to the accumulated field in-place.
// Skips voxels where the intensity difference is below IntensityDifferenceThreshold
// or the denominator is below DenominatorThreshold (numerically unstable region).
__global__ void kernel_warp_and_update(const float* fixed, const float* moving,
                                       const float* gx, const float* gy, const float* gz,
                                       float* dax, float* day, float* daz,
                                       int rows, int cols, int slices) {
    int c = blockIdx.x * blockDim.x + threadIdx.x; // Cols (x)
    int r = blockIdx.y * blockDim.y + threadIdx.y; // Rows (y)
    int s = blockIdx.z * blockDim.z + threadIdx.z; // Slices (z)

    if (r >= rows || c >= cols || s >= slices) return;

    long long idx = (long long)s * rows * cols + (long long)c * rows + r;

    // 1. Read current displacement
    float u_x = dax[idx];
    float u_y = day[idx];
    float u_z = daz[idx];

    // Sampling coordinates: convert 0-based thread index + displacement to 0-based
    // fractional grid coordinates for trilinear interpolation.
    // MATLAB intrinsic coords are 1-based, so the offset cancels:
    //   sample_c = (c + 1 + u_x) - 1 = c + u_x
    float sample_c = c + u_x;
    float sample_r = r + u_y;
    float sample_s = s + u_z;

    // 3. Warp (Trilinear Interpolation)
    float moving_val = sample_vol_trilinear(moving, rows, cols, slices, sample_r, sample_c, sample_s);

    // 4. Compute Update
    float fixed_val = fixed[idx];
    float diff = fixed_val - moving_val; // Fixed - MovingWarped

    // If moving_val is NaN (out of bounds), diff will be NaN.
    
    float g_x = gx[idx];
    float g_y = gy[idx];
    float g_z = gz[idx];
    float gradMagSq = g_x*g_x + g_y*g_y + g_z*g_z;

    float denominator = gradMagSq + diff*diff;

    float du_x = 0.0f;
    float du_y = 0.0f;
    float du_z = 0.0f;

    // Minimum intensity difference to generate a non-zero update (avoids
    // noise-driven updates in flat regions).
    float IntensityDifferenceThreshold = 0.001f;
    // Minimum denominator to avoid division by near-zero values.
    float DenominatorThreshold = 1e-9f;

    // Skip update if: out-of-bounds warp (NaN diff), flat region, or unstable denominator.
    if (denominator >= DenominatorThreshold && fabs(diff) >= IntensityDifferenceThreshold && !isnan(diff)) {
        float factor = diff / denominator;
        du_x = factor * g_x;
        du_y = factor * g_y;
        du_z = factor * g_z;
    }

    // 5. Additive Update
    dax[idx] = u_x + du_x;
    day[idx] = u_y + du_y;
    daz[idx] = u_z + du_z;
}

// 3. Separable Gaussian convolution kernels – one pass per spatial dimension.
// Border condition: replicate (clamp-to-edge), matching MATLAB's imfilter default.

__global__ void kernel_conv_x(const float* in, float* out, const float* kernel, int kRadius, int rows, int cols, int slices) {
    // Convolve along the column dimension (dim2, stride = rows in column-major layout).
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int s = blockIdx.z * blockDim.z + threadIdx.z;

    if (r >= rows || c >= cols || s >= slices) return;

    long long idx = (long long)s * rows * cols + (long long)c * rows + r;

    float sum = 0.0f;
    for (int k = -kRadius; k <= kRadius; k++) {
        int curC = c + k;
        if (curC < 0) curC = 0;
        if (curC >= cols) curC = cols - 1;

        long long fetchIdx = (long long)s * rows * cols + (long long)curC * rows + r;
        sum += in[fetchIdx] * kernel[k + kRadius];
    }
    out[idx] = sum;
}

__global__ void kernel_conv_y(const float* in, float* out, const float* kernel, int kRadius, int rows, int cols, int slices) {
    // Convolve along the row dimension (dim1, contiguous in column-major layout).
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int s = blockIdx.z * blockDim.z + threadIdx.z;

    if (r >= rows || c >= cols || s >= slices) return;

    long long idx = (long long)s * rows * cols + (long long)c * rows + r;

    float sum = 0.0f;
    for (int k = -kRadius; k <= kRadius; k++) {
        int curR = r + k;
        if (curR < 0) curR = 0;
        if (curR >= rows) curR = rows - 1;

        long long fetchIdx = (long long)s * rows * cols + (long long)c * rows + curR;
        sum += in[fetchIdx] * kernel[k + kRadius];
    }
    out[idx] = sum;
}

__global__ void kernel_conv_z(const float* in, float* out, const float* kernel, int kRadius, int rows, int cols, int slices) {
    // Convolve along the slice dimension (dim3, stride = rows * cols).
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int s = blockIdx.z * blockDim.z + threadIdx.z;

    if (r >= rows || c >= cols || s >= slices) return;

    long long idx = (long long)s * rows * cols + (long long)c * rows + r;

    float sum = 0.0f;
    for (int k = -kRadius; k <= kRadius; k++) {
        int curS = s + k;
        if (curS < 0) curS = 0;
        if (curS >= slices) curS = slices - 1;

        long long fetchIdx = (long long)curS * rows * cols + (long long)c * rows + r;
        sum += in[fetchIdx] * kernel[k + kRadius];
    }
    out[idx] = sum;
}


// --------------------------------------------------------------------------
// MEX Entry Point
// --------------------------------------------------------------------------
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 7) mexErrMsgIdAndTxt("demons3d_cuda:invalidNumInputs", "Expected 7 inputs: moving, fixed, N, sigma, Dax, Day, Daz.");
    
    mxInitGPU();

    // --- Input GPU arrays ---
    const mxGPUArray* mMoving = mxGPUCreateFromMxArray(prhs[0]);
    const mxGPUArray* mFixed  = mxGPUCreateFromMxArray(prhs[1]);

    int   N_iter = (int)mxGetScalar(prhs[2]); // number of iterations
    float sigma = (float)mxGetScalar(prhs[3]); // Gaussian smoothing sigma

    const mxGPUArray* mDax = mxGPUCreateFromMxArray(prhs[4]);
    const mxGPUArray* mDay = mxGPUCreateFromMxArray(prhs[5]);
    const mxGPUArray* mDaz = mxGPUCreateFromMxArray(prhs[6]);

    if (mxGPUGetClassID(mMoving) != mxSINGLE_CLASS ||
        mxGPUGetClassID(mFixed)  != mxSINGLE_CLASS ||
        mxGPUGetClassID(mDax)    != mxSINGLE_CLASS ||
        mxGPUGetClassID(mDay)    != mxSINGLE_CLASS ||
        mxGPUGetClassID(mDaz)    != mxSINGLE_CLASS) {
        mexErrMsgIdAndTxt("demons3d_cuda:typeMismatch", "All gpuArray inputs must be single.");
    }

    // MATLAB gpuArray is column-major: dims = [rows, cols, slices]
    const mwSize* dims = mxGPUGetDimensions(mMoving);
    int rows = (int)dims[0]; 
    int cols = (int)dims[1]; 
    int slices = (int)dims[2];
    long long numel = (long long)rows * cols * slices;

    // Raw device pointers to the (read-only) input arrays
    const float* d_moving = (const float*)mxGPUGetDataReadOnly(mMoving);
    const float* d_fixed  = (const float*)mxGPUGetDataReadOnly(mFixed);

    // Deep-copy the input displacement fields; outputs are modified in-place during iterations.
    mxGPUArray* outDax = mxGPUCopyReal(mDax);
    mxGPUArray* outDay = mxGPUCopyReal(mDay);
    mxGPUArray* outDaz = mxGPUCopyReal(mDaz);

    float* d_dax = (float*)mxGPUGetData(outDax);
    float* d_day = (float*)mxGPUGetData(outDay);
    float* d_daz = (float*)mxGPUGetData(outDaz);

    // Temporary device buffers:
    //   d_gx/y/z  – fixed-image gradients (computed once, reused every iteration)
    //   d_temp1/2 – ping-pong buffers for the three-pass separable Gaussian
    float *d_gx, *d_gy, *d_gz, *d_temp1, *d_temp2;
    cudaMalloc(&d_gx, numel * sizeof(float));
    cudaMalloc(&d_gy, numel * sizeof(float));
    cudaMalloc(&d_gz, numel * sizeof(float));
    cudaMalloc(&d_temp1, numel * sizeof(float));
    cudaMalloc(&d_temp2, numel * sizeof(float));

    // Thread block: 8^3 = 512 threads. Grid covers the full volume.
    // Mapping: blockIdx.x -> cols, blockIdx.y -> rows, blockIdx.z -> slices.
    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
    dim3 grid((cols + block.x - 1) / block.x, 
              (rows + block.y - 1) / block.y, 
              (slices + block.z - 1) / block.z);

    // ----------------------------------------------------------------------
    // 1. Precompute fixed-image gradients (done once outside the main loop)
    // ----------------------------------------------------------------------
    kernel_gradient<<<grid, block>>>(d_fixed, d_gx, d_gy, d_gz, rows, cols, slices);
    cudaCheckError();

    // ----------------------------------------------------------------------
    // 2. Build and upload the 1-D Gaussian kernel
    // ----------------------------------------------------------------------
    // Build a normalised 1-D Gaussian kernel of radius ceil(3*sigma).
    int r = (int)ceil(3.0 * sigma);
    int kSize = 2 * r + 1;
    std::vector<float> h_kernel(kSize);
    float sum = 0.0f;
    for (int i = 0; i < kSize; i++) {
        float x = (float)(i - r);
        h_kernel[i] = expf(-(x*x) / (2.0f * sigma * sigma));
        sum += h_kernel[i];
    }
    for (int i = 0; i < kSize; i++) h_kernel[i] /= sum;

    float* d_kernel;
    cudaMalloc(&d_kernel, kSize * sizeof(float));
    cudaMemcpy(d_kernel, h_kernel.data(), kSize * sizeof(float), cudaMemcpyHostToDevice);

    // ----------------------------------------------------------------------
    // 3. Main iteration loop
    // ----------------------------------------------------------------------
    for (int iter = 0; iter < N_iter; iter++) {

        // A. Fused warp + update: reads d_dax/y/z, writes updated values in-place.
        kernel_warp_and_update<<<grid, block>>>(d_fixed, d_moving, d_gx, d_gy, d_gz,
                                                d_dax, d_day, d_daz, 
                                                rows, cols, slices);
        cudaCheckError();

        // B. Regularise: separable Gaussian smoothing of each displacement component.
        //    Pipeline: component -> temp1 (X pass) -> temp2 (Y pass) -> component (Z pass).
        
        // Smooth Dax
        kernel_conv_x<<<grid, block>>>(d_dax, d_temp1, d_kernel, r, rows, cols, slices);
        kernel_conv_y<<<grid, block>>>(d_temp1, d_temp2, d_kernel, r, rows, cols, slices);
        kernel_conv_z<<<grid, block>>>(d_temp2, d_dax, d_kernel, r, rows, cols, slices);

        // Smooth Day
        kernel_conv_x<<<grid, block>>>(d_day, d_temp1, d_kernel, r, rows, cols, slices);
        kernel_conv_y<<<grid, block>>>(d_temp1, d_temp2, d_kernel, r, rows, cols, slices);
        kernel_conv_z<<<grid, block>>>(d_temp2, d_day, d_kernel, r, rows, cols, slices);

        // Smooth Daz
        kernel_conv_x<<<grid, block>>>(d_daz, d_temp1, d_kernel, r, rows, cols, slices);
        kernel_conv_y<<<grid, block>>>(d_temp1, d_temp2, d_kernel, r, rows, cols, slices);
        kernel_conv_z<<<grid, block>>>(d_temp2, d_daz, d_kernel, r, rows, cols, slices);
        
        cudaCheckError();
    }

    // --- Free temporary device buffers ---
    cudaFree(d_gx);
    cudaFree(d_gy);
    cudaFree(d_gz);
    cudaFree(d_temp1);
    cudaFree(d_temp2);
    cudaFree(d_kernel);

    mxGPUDestroyGPUArray(mMoving);
    mxGPUDestroyGPUArray(mFixed);
    mxGPUDestroyGPUArray(mDax);
    mxGPUDestroyGPUArray(mDay);
    mxGPUDestroyGPUArray(mDaz);

    // --- Return updated displacement fields to MATLAB ---
    plhs[0] = mxGPUCreateMxArrayOnGPU(outDax);
    plhs[1] = mxGPUCreateMxArrayOnGPU(outDay);
    plhs[2] = mxGPUCreateMxArrayOnGPU(outDaz);

    mxGPUDestroyGPUArray(outDax);
    mxGPUDestroyGPUArray(outDay);
    mxGPUDestroyGPUArray(outDaz);
}
