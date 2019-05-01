#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_
#include <mxnet/base.h>
#define TILE_WIDTH 16
namespace mxnet
{
    namespace op
    {
        __global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const in
            t W, const int K)
            {
                __shared__ float shared_mem[48000 /sizeof(float)]; // num channels, amount in block in both row and col directions
                int n, m, c, p, q;
                float output;
                const int H_out = H - K + 1;
                const int W_out = W - K + 1;
                //helps us index the pointers
                #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
                #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
                #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
                const int block_size = TILE_WIDTH + K - 1;
                const int mask_radius = (K - 1) / 2;
                const int ty = threadIdx.y;
                const int tx = threadIdx.x;
                const int block_y = blockIdx.z / block_size;
                const int block_x = blockIdx.z % block_size;
                const int row_o = block_y * TILE_WIDTH + ty;
                const int col_o = block_x * TILE_WIDTH + tx;
                const int row_i = row_o - mask_radius;
                const int col_i = col_o - mask_radius;
                output = 0.0f;
                n = blockIdx.x; // idx of batch
                m = blockIdx.y; // idx of features
                // load elements into shared memory one for each thread and each thread must do all channels of its pixel
                for (c = 0; c < C; c++) {
                    if (row_i >= 0 && row_i < H
                        && col_i >= 0 && col_i < W)
                        shared_mem[c * block_size * block_size + ty * block_size + tx] = x4d(n, c, row_i, col_i);
                        else
                        shared_mem[c * block_size * block_size + ty * block_size + tx] = 0.0f;
                    }
                    // do computation by going through channels, and the width and height of the kernel
                    if (ty < TILE_WIDTH && tx < TILE_WIDTH) {
                        for (c = 0; c < C; c++) {
                            for (p = 0; p < K; p++) {
                                for(q = 0; q < K; q++) {
                                    output += k4d(m, c, p, q) * shared_mem[c * block_size * block_size + (ty + p) * block_size + (tx + q)];
                                }

                            }
                        }
                    }
                    __syncthreads();
                    if (row_o < H_out && col_o < W_out)
                    y4d(n, m, row_o, col_o) = output;
                    #undef y4d
                    #undef x4d
                    #undef k4d
                }
                /*
                This function is called by new-inl.h
                Any code you write should be executed by this function.
                For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
                */
                template <>
                void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, fl
                    oat> &w)
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
                        int W_grid =ceil(W / (float)TILE_WIDTH); // number of horizontal tiles per output map
                        int H_grid = ceil(H / (float)TILE_WIDTH); // number of vertical tiles per output map
                        int Z = H_grid * W_grid;
                        int input_block_size = TILE_WIDTH + K - 1;
                        dim3 blockDim(input_block_size, input_block_size, 1);
                        dim3 gridDim(B, M, Z); //num of output images, number of output feature maps, total tiles
                        // Call the kernel
                        forward_kernel<<<gridDim, blockDim>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);
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
