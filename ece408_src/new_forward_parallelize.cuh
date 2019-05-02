#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#define BLOCK_SIZE 1024
#define TILE_WIDTH 24

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

__global__ void matrixMultiply(float *A, float *B, float *Y, int numAColumns, int numCRows, int numCColumns, int M, int C, int H, int W, int K, int H_out, int W_out)

{
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
  // int numAColumns = C*K*K;
  // int numCColumns = H_out*W_out;
  B += blockIdx.z * C * H * W;
  Y += blockIdx.z * M * numCColumns;

  __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
  __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int rowIdx = blockIdx.y * TILE_WIDTH + ty;
  int colIdx = blockIdx.x * TILE_WIDTH + tx;

  float pValue = 0;
  int q, p, c_out, w, h;

  for (int tileIdx = 0; tileIdx < ceil(float(numAColumns)/TILE_WIDTH);  ++tileIdx) {

    if (rowIdx < numCRows && tileIdx*TILE_WIDTH + tx < numAColumns)
      tileA[ty][tx] = A[rowIdx * numAColumns+tileIdx*TILE_WIDTH + tx];
    else
      tileA[ty][tx] = 0;

    int matrix_row = tileIdx * TILE_WIDTH + ty;
    if (colIdx < numCColumns && matrix_row < numAColumns) {
      q = matrix_row % K;
      matrix_row /= K;
      p = matrix_row % K;
      c_out = matrix_row / K;
      w = colIdx % W_out;
      h = colIdx / W_out;
      //populate tileB similar to how convolution layer
      tileB[ty][tx] = B[c_out * (H * W) + (h+p) * (W) + w+q];
    }
    else
      tileB[ty][tx] = 0;

    __syncthreads();

    for (int k = 0; k < TILE_WIDTH; k++)
       pValue += tileA[ty][k] * tileB[k][tx];

    __syncthreads();
  }

  if ((rowIdx < numCRows) && (colIdx < numCColumns)) {
    Y[rowIdx*numCColumns+colIdx] = pValue;
  }
}

/*
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &k)
{

    // Use mxnet's CHECK_EQ to do assertions.
    // Remove this assertion when you do your implementation!
    // CHECK_EQ(0, 1) << "Remove this line and replace with your implementation";

    // Extract the tensor dimensions into B,M,C,H,W,K
    // ...
    const int B = x.shape_[0];
    const int M = y.shape_[1];
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = k.shape_[3];
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    int gridDimY = ceil(float(M)/TILE_WIDTH)
    int gridDimX = ceil(float(H_out*W_out)/TILE_WIDTH);
    dim3 gridDim (gridDimX, gridDimY, B);
    dim3 blockDim (TILE_WIDTH, TILE_WIDTH, 1);
    int W_unroll = C * K * K;
    int H_unroll = H_out * W_out;

    matrixMultiply<<<gridDim, blockDim>>>(k.dptr_, x.dptr_, y.dptr_, W_unroll, M, H_unroll, M, C, H, W, K, H_out, W_out);

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
