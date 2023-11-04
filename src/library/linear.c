#include <stdio.h>
#include "linear.h"
void matmul(float * output, float * x, float * weight, int m, int n);



LinearModule * create_linear_module(int input_size, int output_size);
void load_linear_module(LinearModule * linear_module, char * path);
void free_linear_module(LinearModule * linear_module);
void forward_linear_module(LinearModule * linear_module, float * input,float * output);


LinearModule * create_linear_module(int input_size, int output_size) {
    LinearModule * linear_module = malloc(sizeof(LinearModule));
    linear_module->input_size = input_size;
    linear_module->output_size = output_size;
    linear_module->weight = malloc(sizeof(float) * output_size * input_size);
    linear_module->bias = malloc(sizeof(float) * output_size);
    return linear_module;
}
void load_linear_module(LinearModule * linear_module, char * path){
    FILE * fp = fopen(path, "rb");
    if (fp == NULL) {
        printf("Error: open file %s failed.\n", path);
        exit(1);
    }
    int input_size = linear_module->input_size;
    int output_size = linear_module->output_size;
    fread(linear_module->weight, sizeof(float), input_size * output_size, fp);
    fread(linear_module->bias, sizeof(float), output_size, fp);
    fclose(fp);
}
void free_linear_module(LinearModule * linear_module) { 
    free(linear_module->weight);
    free(linear_module->bias);
    free(linear_module);
}
void forward_linear_module(LinearModule * linear_module, float * input,float * output) {

    matmul(output, input, linear_module->weight, linear_module->output_size, linear_module->input_size);
    for (int i = 0; i < linear_module->output_size; i++) {
        output[i] += linear_module->bias[i];
    }
    return output;
}

void matmul(float * output, float * x, float * weight, int m, int n){
    // weight: m * n; x : n; output: m * 1
    // assert output is zero initailly
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++){
            output[i] += x[j] * weight[i * n + j];
        }
    }
    return;
}

void * build_linear_module(char * path,int input_size,int output_size){
    LinearModule * linear_module = create_linear_module(input_size, output_size);
    load_linear_module(linear_module, path);
    return linear_module;
}

float * test_linear_module(void* lm,float * input){
    LinearModule * linear_module = lm;
    float * output = (float *) calloc(sizeof(float) , linear_module->output_size);
    forward_linear_module(linear_module, input, output);
    return output;
}

void free_output(float * p){
    free(p);
}
// int main(int argc, char * argv[]){
//     char * path = "/home/xs/Code/Python/MachineLearning/base_module/bin/LinearModule.bin";
//     int input_size = 5;
//     int output_size = 10;

//     LinearModule * linear_module = create_linear_module(input_size, output_size);
//     load_linear_module(linear_module, path);
//     float input[5] = {1.f,2.f,3.f,4.f,5.f};
//     float * output = (float *) calloc(sizeof(float) , output_size);
//     forward_linear_module(linear_module, input, output);
//     for(int i = 0; i < input_size; i++){
//         printf("%f ", input[i]);
//     }
//     printf("\n");
//     for(int i = 0; i < output_size; i++){
//         printf("%f ", output[i]);
//     }
//     printf("\n");
//     // free(input);
//     free(output);
//     free_linear_module(linear_module);
//     return 0;
// }

void test_input(float * input,int input_size){
    for (int i = 0; i < input_size; i++){
        printf("%f ", input[i]);
    }
    printf("\n");
}