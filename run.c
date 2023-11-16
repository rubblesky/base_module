#include "src/library/linear.h"
#include "src/library/conv.h"
#include "src/library/batchnorm.h"
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char * argv[]){


    run_linear_test(argc,argv);
    return 0;
    
}
    // char * path = "/home/xs/Code/Python/MachineLearning/base_module/bin/LinearModule.bin";
    // int input_size = 5;
    // int output_size = 10;
    // LinearModule * lm;
    // lm = build_linear_module(path, input_size, output_size);
    // float input[5] = {1.f,2.f,3.f,4.f,5.f};
    // float * output = test_linear_module(lm, input);
    // // LinearModule * linear_module = create_linear_module(input_size, output_size);
    // // load_linear_module(linear_module, path);
    // // float input[5] = {1.f,2.f,3.f,4.f,5.f};
    // // float * output = (float *) calloc(sizeof(float) , output_size);
    // // forward_linear_module(linear_module, input, output);
    // for(int i = 0; i < input_size; i++){
    //     printf("%f ", input[i]);
    // }
    // printf("\n");
    // for(int i = 0; i < output_size; i++){
    //     printf("%f ", output[i]);
    // }
    // printf("\n");
    // // free(input);
    // // free(output);
    // // free_linear_module(linear_module);

// #include <stdio.h>
// #include <stdlib.h>

// void conv2d(float* input, float* kernel, float* output, int N, int C, int H, int W, int D, int K, int P) {
//     for (int n = 0; n < N; ++n) {
//         for (int d = 0; d < D; ++d) {
//             for (int h = -P; h < H - K + 1 + P; ++h) {
//                 for (int w = -P; w < W - K + 1 + P; ++w) {
//                     float sum = 0.0f;
//                     for (int c = 0; c < C; ++c) {
//                         for (int kh = 0; kh < K; ++kh) {
//                             for (int kw = 0; kw < K; ++kw) {
//                                 int x = h + kh;
//                                 int y = w + kw;
//                                 if (x >= 0 && x < H && y >= 0 && y < W) {
//                                     sum += input[n*C*H*W + c*H*W + x*W + y] * kernel[d*C*K*K + c*K*K + kh*K + kw];
//                                 }
//                             }
//                         }
//                     }
//                     if (h >= 0 && w >= 0) {
//                         output[n*D*(H-K+1+2*P)*(W-K+1+2*P) + d*(H-K+1+2*P)*(W-K+1+2*P) + h*(W-K+1+2*P) + w] = sum;
//                     }
//                 }
//             }
//         }
//     }
// }

// int main() {
//     // 假设我们有一个形状为(N, C, H, W)的输入数据和一个形状为(D, C, K, K)的卷积核
//     int N = 1, C = 1, H = 5, W = 5, D = 1, K = 3, P = 1;
//     float* input = (float*)malloc(N * C * H * W * sizeof(float));
//     float* kernel = (float*)malloc(D * C * K * K * sizeof(float));
//     float* output = (float*)malloc(N * D * (H - K + 1 + 2*P) * (W - K + 1 + 2*P) * sizeof(float));

//     // 初始化输入数据和卷积核
//     for (int i = 0; i < N*C*H*W; ++i) input[i] = i % 10;
//     for (int i = 0; i < D*C*K*K; ++i) kernel[i] = i % 10;

//     // 执行卷积操作
//     conv2d(input, kernel, output, N, C, H, W, D, K, P);

//     // 打印输出数据
//     for (int i = 0; i < N*D*(H-K+1+2*P)*(W-K+1+2*P); ++i) printf("%f ", output[i]);
//     printf("\n");

//     // 释放内存
//     free(input);
//     free(kernel);
//     free(output);

//     return 0;
// }
