#ifndef LAYERNORM_H
#define LAYERNORM_H
#include <stdio.h>
#include "tensor.h"
#define MAX_DIM 10
typedef struct LayerNormModule {
    int dim;
    int *shape;
    int num_features;
    float eps;
    float *weight;
    float *bias;
}LayerNormModule;

//LayerNormModule * create_layernorm_module(int dim,int shape[],float eps);
//void load_layernorm_module(LayerNormModule *module,FILE *fp);

LayerNormModule * build_layernorm_module(char * path);
void forward_layernorm_module(LayerNormModule * module,Tensor * input,Tensor * output);
void free_layernorm_module(LayerNormModule *module);

Tensor * create_layernorm_output(LayerNormModule * module, Tensor * input);
void free_layernorm_output(Tensor *output);
int run_layernorm_test(int argc, char * argv[]);
#endif