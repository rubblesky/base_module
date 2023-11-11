#ifndef BATCHNORM_H
#define BATCHNORM_H
#include <stdio.h>
typedef struct BatchNormModule
{
    int num_features;
    float eps;
    float * weights;
    float * bias;
    float * running_mean;
    float * running_var;
}BatchNormModule;

BatchNormModule * create_batchnorm_module(int num_features,float eps);
void load_batchnorm_module(BatchNormModule * module,FILE * fp);
void free_batchnorm_module(BatchNormModule * module);
void batchnorm_module_forward(BatchNormModule * module,float * input,float * output,int *input_size,int *output_size);

float * create_batchnorm_output(BatchNormModule * module, int *input_size,int *output_size);
void * free_batchnorm_output(float * output);

BatchNormModule *build_batchnorm_module(char * path);
void * test_batchnorm_module(BatchNormModule * module, float * data,int height,int width);
int run_batchnorm_test(int argc, char * argv[]);
#endif