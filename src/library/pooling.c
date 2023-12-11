#include <stdio.h>
#include <stdlib.h>
#include "pooling.h"
#include <float.h>
float NEGATIVE_INFINITY = -FLT_MAX;

PoolingModule * build_pooling_module(char * path){
    FILE * fp = fopen(path,"rb");
    if (fp == NULL) {
        printf("Error: open file %s failed.\n", path);
        exit(-1);
    }
    int size = 0;
    int kernel_size[2];
    int stride[2];
    int padding[2];
    size += fread(kernel_size, sizeof(int), 2, fp);
    size += fread(stride, sizeof(int), 2, fp);
    size += fread(padding, sizeof(int), 2, fp);
    if(size != 2 + 2 + 2){
        printf("Error: read file %s failed.\n", path);
        exit(-1);
    }
    PoolingModule * module = create_pooling_module(kernel_size,stride,padding);
    fclose(fp);
    return module;
}
PoolingModule * create_pooling_module(int kernel_size[2],int stride[2], int padding[2]){
    PoolingModule * module = malloc(sizeof (PoolingModule));
    module->kernel_size[0] = kernel_size[0];
    module->kernel_size[1] = kernel_size[1];
    module->stride[0] = stride[0];
    module->stride[1] = stride[1];
    module->padding[0] = padding[0];
    module->padding[1] = padding[1];
    return module;
}
void forward_maxpooling_module(PoolingModule * module,Tensor * input,Tensor * output){
    int input_size[2] = {input->shape[input->num_dim - 2],input->shape[input->num_dim - 1]};
    int output_size[2] = {output->shape[output->num_dim - 2],output->shape[output->num_dim - 1]};
    int channels = input->shape[input->num_dim - 3];

    for(int c = 0;c < channels;c++){
        float * output_cpos = output->data + c * output_size[0] * output_size[1];
        float* input_cpos = input->data + c * input_size[0] * input_size[1];
        for(int i = 0;i < output_size[0];i++){
            for(int j = 0;j < output_size[1];j++){
                int in_x = i * module->stride[0] - module->padding[0];
                int in_y = j * module->stride[1] - module->padding[1];
                float * output_pos = output_cpos + i * output_size[1] + j;
                float max = NEGATIVE_INFINITY;
                for(int k1 = 0; k1 < module->kernel_size[0];k1++){
                    for(int k2 = 0; k2 < module->kernel_size[1];k2++){
                        int x = in_x + k1;
                        int y = in_y + k2;
                        if(x < 0 || y < 0 || x >= input_size[0] || y >= input_size[1]) continue;
                        max = input_cpos[x * input_size[1] + y] > max ? input_cpos[x * input_size[1] + y]:max;
                    }
                }
                *output_pos = max;
            }
        }
    }

    return;
}
void free_pooling_module(PoolingModule *module){
    free(module);
}

Tensor * create_pooling_output(PoolingModule * module, Tensor * input){
    int input_size[2] = {input->shape[input->num_dim - 2],input->shape[input->num_dim - 1]};
    int output_size[input->num_dim];
    for(int i = 0; i < input->num_dim - 2;i++){
        output_size[i] = input->shape[i];
    }
    output_size[input->num_dim - 2] = (input_size[0] + module->padding[0] * 2 - module->kernel_size[0]) / module->stride[0] + 1;
    output_size[input->num_dim - 1] = (input_size[1] + module->padding[1] * 2 - module->kernel_size[1]) / module->stride[1] + 1;
    Tensor * output = Tensor_init(input->num_dim,output_size);
    return output;
}
void free_pooling_output(Tensor *output){
    free_tensor(output);
}
int run_pooling_test(int argc, char * argv[]){
    char *path = "/home/xs/Code/Python/MachineLearning/base_module/bin/MaxPoolingModule.bin";

    PoolingModule * module = build_pooling_module(path);
    float input_data[18] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18};
    int in_size[3] = {1,3,4};
    Tensor * input = Tensor_new(3,in_size,input_data);


    Tensor * output = create_pooling_output(module, input);
    forward_maxpooling_module(module, input, output);
    print_tensor(output);
    free_pooling_output(output);
    free_pooling_module(module);
}