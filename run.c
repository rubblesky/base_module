#include "src/library/linear.h"
#include <stdio.h>
int main(int argc, char * argv[]){
    char * path = "/home/xs/Code/Python/MachineLearning/base_module/bin/LinearModule.bin";
    int input_size = 5;
    int output_size = 10;
    LinearModule * lm = NULL;
    lm = build_linear_module(path, input_size, output_size);
    float input[5] = {1.f,2.f,3.f,4.f,5.f};
    float * output = test_linear_module(lm, input);
    // LinearModule * linear_module = create_linear_module(input_size, output_size);
    // load_linear_module(linear_module, path);
    // float input[5] = {1.f,2.f,3.f,4.f,5.f};
    // float * output = (float *) calloc(sizeof(float) , output_size);
    // forward_linear_module(linear_module, input, output);
    for(int i = 0; i < input_size; i++){
        printf("%f ", input[i]);
    }
    printf("\n");
    for(int i = 0; i < output_size; i++){
        printf("%f ", output[i]);
    }
    printf("\n");
    // free(input);
    // free(output);
    // free_linear_module(linear_module);
    return 0;

}