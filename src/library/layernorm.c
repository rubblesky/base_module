#include "layernorm.h"
#include <stdio.h>
#include <stdlib.h>

LayerNormModule * create_layernorm_module(int dim,int shape[],float eps){
    LayerNormModule * module = (LayerNormModule *)malloc(sizeof(LayerNormModule));
    module->dim = dim;
    module->eps = eps;
    module->shape = malloc(sizeof(int) * dim);
    int num_features = 1;
    for (int i = 0;i < dim,i++){
        length *= shape[i];
        module->shape[i] = shape[i];
    }
    module->num_features = num_features;
    module->weight = malloc(sizeof(float) * length);
    module->bias = malloc(sizeof(float) * length);
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
    size += fread(shape,sizeof(int),dim,fp)
    if(size != 2 && dim < MAX_DIM){
        printf("Error: read file %s failed.\n",path);
        exit(-1);
    }
    LayerNormModule * module = create_layernorm_module(dim,eps);
    load_layernorm_module(module,fp);
    fclose(fp);
    return module;
}