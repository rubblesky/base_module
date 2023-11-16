#include "layernorm.h"
#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

LayerNormModule * create_layernorm_module(int dim,int shape[],float eps){
    LayerNormModule * module = (LayerNormModule *)malloc(sizeof(LayerNormModule));
    module->dim = dim;
    module->eps = eps;
    module->shape = malloc(sizeof(int) * dim);
    int num_features = 1;
    for (int i = 0;i < dim;i++){
        num_features *= shape[i];
        module->shape[i] = shape[i];
    }
    module->num_features = num_features;
    module->weight = malloc(sizeof(float) * num_features);
    module->bias = malloc(sizeof(float) * num_features);
    return module;
}
void load_layernorm_module(LayerNormModule *module,FILE *fp){
    int size = 0;
    size += fread(module->weight,sizeof(float),module->num_features,fp);
    size += fread(module->bias,sizeof(float),module->num_features,fp);
    if(size != 2* module->num_features){
        printf("load layernorm module error \n");
        exit(-1);
    }
}
void free_layernorm_module(LayerNormModule *module){
    free(module->bias);
    free(module->shape);
    free(module->weight);
    free(module);
}

LayerNormModule * build_layernorm_module(char * path){
    FILE * fp = fopen(path,"rb");
    if(fp == NULL){
        printf("Error: open file %s filed.\n",path);
        exit(-1);
    }
    int dim;
    float eps;
    int shape[MAX_DIM];
    int size = 0;
    size += fread(&dim,sizeof(int),1,fp);
    size += fread(&eps,sizeof(float),1,fp);
    size += fread(shape,sizeof(int),dim,fp);
    if(size != 2 && dim < MAX_DIM){
        printf("Error: read file %s failed.\n",path);
        exit(-1);
    }
    LayerNormModule * module = create_layernorm_module(dim,shape,eps);
    load_layernorm_module(module,fp);
    fclose(fp);
    return module;
}
void forward_layernorm_module(LayerNormModule * module,Tensor * input,Tensor * output){
    if(input->num_features % module->num_features != 0){
        printf("Error: The number of features of the input must be divisible by the number of features of the module\n");
        exit(-1);
    }
    int len = input->num_features / module->num_features;
    for(int i = input->num_dim - module->dim; i < input->num_dim;i--){
        output->shape[i] = module->shape[i];
        if(output->shape[i] != input->shape[i]){
            printf("Error: The last %d dimensions of the input must be consistent with the dimensions of the module\n",module->dim);
        }
    }
    for(int i = 0;i < input->num_dim - module->dim;i++){
        output->shape[i] = input->shape[i];
    }
    for(int i = 0;i < len;i++){
        float mean = 0;
        float vars = 0;
        for(int j = 0;j < module->num_features;j++){
            mean += input->data[i * module->num_features + j];
        }
        mean = mean / module->num_features;
        for(int j = 0;j < module->num_features;j++){
            vars += (input->data[i * module->num_features + j] - mean) * (input->data[i * module->num_features + j] - mean);
        }
        vars = vars / module->num_features;
        for(int j = 0;j < module->num_features;j++){
            output->data[i * module->num_features + j] = (input->data[i * module->num_features + j] - mean) / sqrtf(vars + module->eps) * module->weight[j] + module->bias[j];
        }
    }
}