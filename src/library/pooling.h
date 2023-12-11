#ifndef POOLING_H
#define POOLING_H
#include "tensor.h"
typedef struct PoolingModule {
    int kernel_size[2];
    int stride[2];
    int padding[2];
}PoolingModule;


PoolingModule * build_pooling_module(char * path);
void forward_maxpooling_module(PoolingModule * module,Tensor * input,Tensor * output);
void free_pooling_module(PoolingModule *module);

PoolingModule * create_pooling_module(int kernel_size[2],int stride[2], int padding[2]);
Tensor * create_pooling_output(PoolingModule * module, Tensor * input);
void free_pooling_output(Tensor *output);
int run_pooling_test(int argc, char * argv[]);
#endif POOLING_H
