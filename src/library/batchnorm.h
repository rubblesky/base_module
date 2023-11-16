#ifndef BATCHNORM_H
#define BATCHNORM_H
#include <stdio.h>
#include "tensor.h"
typedef struct BatchNormModule
{
    int num_features;
    float eps;
    float * weights;
    float * bias;
    float * running_mean;
    float * running_var;
}BatchNormModule;

//BatchNormModule * create_batchnorm_module(int num_features,float eps);
//void load_batchnorm_module(BatchNormModule * module,FILE * fp);

BatchNormModule *build_batchnorm_module(char * path);
void forward_batchnorm_module(BatchNormModule * module,Tensor * input,Tensor * output);
void free_batchnorm_module(BatchNormModule * module);

Tensor * create_batchnorm_output(BatchNormModule * module, Tensor * input);
void free_batchnorm_output(Tensor * output);




int run_batchnorm_test(int argc, char * argv[]);
#endif