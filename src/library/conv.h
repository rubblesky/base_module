#ifndef CONV_H
#define CONV_H
#include "tensor.h"
#include<stdio.h>
typedef struct {
    int in_channels;
    int out_channels;
    int kernel_size[2];
    int stride[2];
    int padding[2];
    int dilation[2];
    int groups;
    float * conv_weights;
    float * conv_bias;
    float * bn_weights;
    float * bn_bias;
} ConvModule;
ConvModule * create_conv_module(int in_channels, 
                                int out_channels, 
                                int *kernel_size, 
                                int *stride, 
                                int *padding, 
                                int *dilation,
                                int groups);
void load_conv_module(ConvModule * module, FILE *fp);
void free_conv_module(ConvModule * module);
ConvModule *build_conv_module(char * path);

Tensor * create_conv_output(ConvModule * module, Tensor * input);
void free_conv_output(Tensor * output);
void forward_conv_module(ConvModule * module, Tensor *input, Tensor *output);


int run_conv_test(int argc, char * argv[]);
#endif
