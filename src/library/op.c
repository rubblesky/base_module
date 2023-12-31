#include "op.h"
void float_matmul(float * output, float * x, float * weight, int m, int n){
    // weight: m * n; x : n; output: m * 1

    for(int i = 0;i < m;i++){
        output[i] = 0;
    }
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++){
            output[i] += x[j] * weight[i * n + j];
        }
    }
    return;
}
void float_add(float *output,float * input, int n){
    for (int i = 0; i < n; i++){
        output[i] += input[i];
    }
}

