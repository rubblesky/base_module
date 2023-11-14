#ifndef LINEAR_H
#define LINEAR_H
#include "tensor.h"
#include <stdio.h>
typedef struct LinearModule
{
    float * weight;
    float * bias;
    int in_features;
    int out_features;
}LinearModule;
LinearModule * create_linear_module(int input_size, int output_size) ;
void load_linear_module(LinearModule * linear_module, FILE * fp);
void free_linear_module(LinearModule * linear_module);
void forward_linear_module(LinearModule * linear_module, Tensor * input,Tensor * output);

void * build_linear_module(char * path);
Tensor * create_linear_output(LinearModule * module, Tensor * input);
void free_linear_output(Tensor * output);
float * test_linear_module(void* lm,float * input);

void test_input(float * input,int input_size);
int run_linear_test(int argc, char * argv[]);
#endif