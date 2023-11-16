#include "batchnorm.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
BatchNormModule * create_batchnorm_module(int num_features,float eps){
    BatchNormModule * module = malloc(sizeof(BatchNormModule));
    module->num_features = num_features;
    module->eps = eps;
    module->weights = malloc(sizeof(float) * num_features);
    module->bias = malloc(sizeof(float) * num_features);
    module->running_mean = malloc(sizeof(float) * num_features);
    module->running_var = malloc(sizeof(float) * num_features);
    return module;
}
void load_batchnorm_module(BatchNormModule * module,FILE * fp){
    int size = 0;
    size += fread(module->weights,sizeof(float) , module->num_features,fp);
    size += fread(module->bias,sizeof(float) , module->num_features,fp);
    size += fread(module->running_mean,sizeof(float) , module->num_features,fp);
    size += fread(module->running_var,sizeof(float) , module->num_features,fp);
    if (size != module->num_features * 4){
        printf("load batchnorm module error \n");
        exit(-1);
    }
}
void free_batchnorm_module(BatchNormModule *module){
    free(module->weights);
    free(module->bias);
    free(module->running_mean);
    free(module->running_var);
    free(module);
}

BatchNormModule * build_batchnorm_module(char *path){
    FILE * fp = fopen(path, "rb");
    if (fp == NULL){
        printf("Error: open file %s failed.\n", path);
        exit(-1);
    }
    int num_features;
    float eps;
    int size = 0;
    size += fread(&num_features,sizeof(int),1,fp);
    size += fread(&eps,sizeof(float),1,fp);
    if(size != 2){
        printf("Error: read file %s failed.\n", path);
        exit(-1);
    }
    BatchNormModule * module = create_batchnorm_module(num_features,eps);
    load_batchnorm_module(module,fp);
    fclose(fp);
    return module;
    
}

void forward_batchnorm_module(BatchNormModule * module,Tensor * input,Tensor * output){
//    if( output_size[0] != input_size[0] ||output_size[1] != input_size[1]){
//        printf("input size and output size are inconsistent in batchnorm\n");
//        exit(-1);
//    }
    int height = input->shape[input->num_dim - 2];
    int width = input->shape[input->num_dim - 1];
    for(int c = 0; c < module->num_features;c++){
        float * input_cpos = input->data + c * height * width;
        float * output_cpos = output->data + c * height * width;
        for(int i = 0;i < height;i++){
            for(int j = 0;j < width;j++){
                output_cpos[i * width + j] = (input_cpos[i * width + j] - module->running_mean[c]) * module->weights[c] / sqrtf(module->running_var[c] + module->eps) + module->bias[c];
            }
        }
    }
}
Tensor * create_batchnorm_output(BatchNormModule * module, Tensor * input){
    Tensor * output = Tensor_like(input);
    return output;
}
void  free_batchnorm_output(Tensor * output){
    free_tensor(output);
}




int run_batchnorm_test(int argc, char * argv[]){
    char * path = "/home/xs/Code/Python/MachineLearning/base_module/bin/BatchNormModule.bin";
    BatchNormModule * module = build_batchnorm_module(path);
    float input_data[9] = {0,1,2,3,4,5,6,7,8 };
    int input_shape[2] = {3,3};
    Tensor * input = Tensor_new(2,input_shape,input_data);
    Tensor * output = create_batchnorm_output(module,input);
    forward_batchnorm_module(module,input,output);
    print_tensor(output);
    free_batchnorm_module(module);
    free_tensor(input);
    free_batchnorm_output(output);
    
}