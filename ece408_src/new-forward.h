
#ifndef MXNET_OPERATOR_NEW_FORWARD_H_
#define MXNET_OPERATOR_NEW_FORWARD_H_

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{


template <typename cpu, typename DType>
void forward(mshadow::Tensor<cpu, 4, DType> &y, const mshadow::Tensor<cpu, 4, DType> &x, const mshadow::Tensor<cpu, 4, DType> &k)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    The code in 16 is for a single image.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct, not fast (this is the CPU implementation.)
    */

    const int B = x.shape_[0]; // images in the batch
    const int M = y.shape_[1]; // features to extract
    const int C = x.shape_[1]; // channels for that one element (ex colors)
    const int H = x.shape_[2]; // height of image
    const int W = x.shape_[3]; // width of image
    const int K = k.shape_[3]; // width of mask (mask is square)

    for (int b = 0; b < B; ++b) {
        
        for (int m = 0; m < M; m++) {
            // H - K + 1 is convolution size
            for (int h = 0; h < H - K + 1; h++) {
                // W - K + 1 is convolution size
                for (int w = 0; w < W - K + 1; w++) {
                    // index into image, then feature, and then y and x coords on the image
                    y[b][m][h][w] = 0;
                    // go over all the channels
                    for (int c = 0; c < C; c++) {
                        // finally compute this one mask. Jeeze
                        for (int p = 0; p < K; p++) {
                            for (int q = 0; q < K; q++) {
                                // convolve with that mask
                                y[b][m][h][w] += x[b][c][h + p][w + q] * k[m][c][p][q];
                            }
                        }
                    }
                }
            }
        }
    }

}
}
}

#endif
