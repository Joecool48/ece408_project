
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>
#define BLOCK_WIDTH 16
#define BLOCK_SIZE 256
#define TILE_WIDTH 32
namespace mxnet
{
namespace op
{
__global__ void matrixMultiplyShared(float *A, float *B, float *C, int numAColumns, int numCRows, int numCColumns)
{

  __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
  __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];
  int rowIdx = blockIdx.y * blockDim.y + tIdx.y;
  int colIdx = blockIdx.x * blockDim.x + tIdx.x;

  float Pvalue = 0;

  //populate shared memory
  for (int tileIdx = 0; tileIdx < ceil(float(numAColumns)/TILE_WIDTH); tileIdx++) {

    int col = tileIdx*TILE_WIDTH+tIdx.x;
    if (col < numAColumns)
      tileA[tIdx.y][tIdx.x] = A[rowIdx*numAColumns+col];
    else
      tileA[tIdx.y][tIdx.x] = 0;

    int row = tileIdx*TILE_WIDTH+tIdx.y;
    if (row < numAColumns)
      tileB[tIdx.y][tIdx.x] = B[row*numCColumns + colIdx];
    else
      tileB[tIdx.y][tIdx.x] = 0;


    __syncthreads();
    for (int k = 0; k < TILE_WIDTH; k++)
       Pvalue += tileA[tIdx.y][k]*tileB[k][tIdx.x];

    __syncthreads();
  }

  if ((rowIdx < numCRows) && (colIdx < numCColumns)) {
    C[rowIdx*numCColumns+colIdx] = Pvalue;
  }

}

//implement unrollKernel according to textbook algorithm
__global__ void unrollKernel(float* X_unrolled, int size, float* X, int C, int K, int H, int W) {
    int tIdx = blockDim.x*blockIdx.x + tIdx.x;
    if (tIdx < size){
        int H_out = H - K + 1;
        int W_out = W - K + 1;
        int W_unroll = H_out * W_out;
        int c = tIdx / W_unroll;
        int s = tIdx % W_unroll;
        int q = c % K;
        c /= K;
        int p = c % K;
        int c_out = c / K;
        int w = s % W_out;
        int h = s / W_out;
        X_unrolled[tIdx] = X[(c_out) * (H * W) + (h+p) * (W) + w+q];
    }
}

//CPU functions that call GPU kernels
void unroll_gpu(int C, int H, int W, int K, int size, float* X, float* X_unrolled){
    int gridDim = ceil(float(size)/BLOCK_SIZE);
    unrollKernel<<<gridDim, BLOCK_SIZE>>>(X_unrolled, size, X, C, K, H, W);
}

//general matrix-multiply
void gemm(int H_unroll, int M, int W_unroll, float *X_unrolled, float* kernel, float* Y){
    dim3 gridDim (ceil(float(H_unroll)/TILE_WIDTH), ceil(float(M)/TILE_WIDTH), 1);
    dim3 blockDim (TILE_WIDTH, TILE_WIDTH, 1);
    //call matrixMultiplyShared kernel (from MP3)
    matrixMultiplyShared<<<gridDim, blockDim>>>(kernel, X_unrolled, Y, W_unroll, M, H_unroll);
}

/*
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{

    // Use mxnet's CHECK_EQ to do assertions.
    // Remove this assertion when you do your implementation!
    //CHECK_EQ(0, 1) << "Remove this line and replace with your implementation";

    // Extract the tensor dimensions into B,M,C,H,W,K

    const int B = x.shape_[0]; //number of output images
    const int M = y.shape_[1]; //number of output feature maps
    const int C = x.shape_[1]; //number of input feature maps
    const int H = x.shape_[2]; //height of output elements
    const int W = x.shape_[3]; //width of output element
    const int K = w.shape_[3]; //dimension of the filters, width and height

    // Set the kernel dimensions
    const int H_out = H - K + 1; // the output after removing the edges
    const int W_out = W - K + 1;

    int W_grid =ceil(W_out/(float)BLOCK_WIDTH); // number of horizontal tiles per output map
    int H_grid = ceil(H_out/(float)BLOCK_WIDTH); // number of vertical tiles per output map

    float* X_unrolled;
    int W_unroll = C * K * K;
    int H_unroll = H_out * W_out;
    int unrolled_size = W_unroll * H_unroll;
    //allocate memory for unrolled
    cudaMalloc(&X_unrolled, sizeof(float) * unrolled_size);


    // Call the kernel
    for (int b = B; b--; ) {
        unroll_gpu(C, H, W, K, unrolled_size, x.dptr_ + b * C * H * W, X_unrolled);
        gemm(H_unroll, M, W_unroll, X_unrolled, w.dptr_, y.dptr_ + b * M * H_unroll);
    }
    cudaFree(X_unrolled);


    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}

/*
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}
}
}

#endif
