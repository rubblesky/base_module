#ifndef LAYERNORM_H
#define LAYERNORM_H
#include <stdio.h>
#define MAX_DIM 10
typedef struct LayerNormModule {
    int dim;
    int *shape;
    int num_features;
    float eps;
    float *weight;
    float *bias;
}LayerNormModule;

LayerNormModule * create_layernorm_module(int dim,int shape[],float eps);
void load_layernorm_module(LayerNormModule *module,FILE *fp);
void free_layernorm_module(LayerNormModule *module);

LayerNormModule * build_layernorm_module(char * path);

float * create_layernorm_output(LayerNormModule * module,int * input_size);


#endif