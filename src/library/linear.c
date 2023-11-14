#include <stdio.h>
#include <stdlib.h>
#include "linear.h"
#include "op.h"
#include "tensor.h"


LinearModule * create_linear_module(int in_features, int out_features) {
    LinearModule * linear_module = malloc(sizeof(LinearModule));
    linear_module->in_features = in_features;
    linear_module->out_features = out_features;
    linear_module->weight = malloc(sizeof(float) * out_features * in_features);
    linear_module->bias = malloc(sizeof(float) * out_features);
    return linear_module;
}
void load_linear_module(LinearModule * linear_module, FILE * fp){
    int in_features = linear_module->in_features;
    int out_features = linear_module->out_features;
    fread(linear_module->weight, sizeof(float), in_features * out_features, fp);
    fread(linear_module->bias, sizeof(float), out_features, fp);

}
void free_linear_module(LinearModule * linear_module) { 
    free(linear_module->weight);
    free(linear_module->bias);
    free(linear_module);
}
void forward_linear_module(LinearModule * module, Tensor * input,Tensor * output) {
    matmul(output->data, input->data, module->weight, module->out_features, module->in_features);
    add(output->data,module->bias, module->out_features);
}

Tensor * create_linear_output(LinearModule * module, Tensor * input){
    int shape[MAX_DIM];
    int num_dim = 0;
    for(int i = 0;i < input->num_dim;i++){
        shape[i] = input->shape[i];
    }
    shape[input->num_dim - 1] = module->out_features;
    num_dim  = input->num_dim;
    Tensor * output = Tensor_init(num_dim,shape);
    return output;
}

void free_linear_output(Tensor * output){
    free_tensor(output);
}

LinearModule * build_linear_module(char * path){
    FILE * fp = fopen(path,"rb");
    if (fp == NULL) {
        printf("Error: open file %s failed.\n", path);
        exit(1);
    }
    int in_features,out_features;
    int size = 0;
    size += fread(&in_features,sizeof (int),1,fp);
    size += fread(&out_features,sizeof(int),1,fp);
    if(size != 1+1){
        printf("Error: read file %s failed.\n", path);
        exit(-1);
    }
    LinearModule * linear_module = create_linear_module(in_features, out_features);
    load_linear_module(linear_module, fp);
    fclose(fp);
    return linear_module;
}



int run_linear_test(int argc, char * argv[]){
    char * path = "/home/xs/Code/Python/MachineLearning/base_module/bin/LinearModule.bin";

    LinearModule * linear_module = build_linear_module(path);
    float input_data[5] = {1.f,2.f,3.f,4.f,5.f};
    int shape[1] = {5};
    Tensor *input = Tensor_new(1,shape,input_data);
    Tensor * output = create_linear_output(linear_module,input);
    forward_linear_module(linear_module, input, output);

    printf("input : \n");
    print_tensor(input);
    printf("\n");
    printf("output:\n");
    print_tensor(output);

    free_tensor(input);
    free_linear_output(output);
    free_linear_module(linear_module);
    return 0;
}

