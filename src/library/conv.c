#include <stdio.h>
#include <stdlib.h>
#include "conv.h"
#include "op.h"
#include "data.h"
// void matmul(float * output, float * x, float * weight, int m, int n);
// void add(float *output,float * input, int n);
ConvModule * create_conv_module(int in_channels, 
                                int out_channels, 
                                int *kernel_size, 
                                int *stride, 
                                int *padding, 
                                int *dilation,
                                int groups){

    ConvModule * module = (ConvModule *)malloc(sizeof(ConvModule));
    module->in_channels = in_channels;
    module->out_channels = out_channels;
    module->kernel_size[0] = kernel_size[0];module->kernel_size[1] = kernel_size[1];
    module->stride[0] = stride[0];module->stride[1] = stride[1];
    module->padding[0] = padding[0];module->padding[1] = padding[1];
    module->dilation[0] = dilation[0];module->dilation[1] = dilation[1];
    module->groups = groups;
    module->conv_weights = (float *)malloc(out_channels * in_channels * kernel_size[0] * kernel_size[1] * sizeof(float));
    module->conv_bias = (float *)malloc(out_channels * sizeof(float));
    module->bn_weights = (float *)malloc(out_channels * sizeof(float));
    module->bn_bias = (float *)malloc(out_channels * sizeof(float));
    return module;
}

void load_conv_module(ConvModule * module, FILE *fp){
    int size = 0;
    size += fread(module->conv_weights, sizeof(float), module->out_channels * module->in_channels * module->kernel_size[0] * module->kernel_size[1], fp);
    size += fread(module->conv_bias, sizeof(float), module->out_channels, fp);
    size += fread(module->bn_weights, sizeof(float), module->out_channels, fp);
    size += fread(module->bn_bias, sizeof(float), module->out_channels, fp);
    if(size != module->out_channels * module->in_channels * module->kernel_size[0] * module->kernel_size[1] + module->out_channels + module->out_channels + module->out_channels){
        printf("load conv module error\n");
        exit(-1);
    }
}
void free_conv_module(ConvModule * module){
    free(module->conv_weights);
    free(module->conv_bias);
    free(module->bn_weights);
    free(module->bn_bias);
    free(module);
}
float * create_conv_output(ConvModule * module, int *input_size,int *output_size){
    // float output_size[2];
    output_size[0] = (input_size[0] + module->padding[0] * 2 - module->kernel_size[0]) / module->stride[0] + 1;
    output_size[1] = (input_size[1] + module->padding[1] * 2 - module->kernel_size[1]) / module->stride[1] + 1;
    float * output = (float *)calloc(sizeof(float),output_size[0] * output_size[1] * module->out_channels);
    return output;
}
void free_conv_output(float * output){
    free(output);
}
void conv_module_forward(ConvModule * module, float *input, float *output,int * input_size,int * output_size){
    // int input_x = input_size[0] + module->padding[0] * 2;   //padded height
    // int input_y = input_size[1] + module->padding[1] * 2;   //padded width


    float * out_data = calloc(sizeof(float),module->out_channels);
    if(out_data == NULL){
        perror("malloc out_data failed");
        exit(-1);
    }
    for (int i = 0; i < output_size[0]; i++){
        for(int j = 0; j < output_size[1]; j++){
            int index = i * output_size[1] + j;
            float * output_pos = output + (index) * module->out_channels;
            int in_x = i * module->stride[0] - module->padding[0];  //top of kernel position in input
            int in_y = j * module->stride[1] - module->padding[1];  //left of kernel position in input
            for(int k1 = 0; k1 < module->kernel_size[0];k1++){
                for(int k2 = 0; k2 < module->kernel_size[1];k2++){
                    int x = in_x + k1;
                    int y = in_y + k2;
                    if(x >= input_size[1] || y >= input_size[0] || x < 0 || y < 0) continue;
                    float * input_pos = input + (x * input_size[1] + y) * module->in_channels;
                    matmul(out_data, input_pos, module->conv_weights + k1 * module->kernel_size[1] + k2, module->out_channels,module->in_channels);
                    add(output_pos, out_data, module->out_channels);
                }
            }
            add(output_pos, module->conv_bias, module->out_channels); //add bias

        }
    }
    free(out_data);
    return;
}


ConvModule *build_conv_module(char * path){
    FILE * fp = fopen(path, "rb");
    if (fp == NULL){
        printf("Error: open file %s failed.\n", path);
        exit(-1);
    }
    int in_channels, out_channels;
    int kernel_size[2];
    int stride[2];
    int padding[2];
    int dilation[2];
    int groups;
    int size = 0;
    size += fread(&in_channels, sizeof(int), 1, fp);
    size += fread(&out_channels, sizeof(int), 1, fp);
    size += fread(kernel_size, sizeof(int), 2, fp);
    size += fread(stride, sizeof(int), 2, fp);
    size += fread(padding, sizeof(int), 2, fp);
    size += fread(dilation, sizeof(int), 2, fp);
    size += fread(&groups, sizeof(int), 1, fp);
    if(size != 1 + 1 + 2 + 2 + 2 + 2 + 1){
        printf("Error: read file %s failed.\n", path);
        exit(1);
    }
    ConvModule * conv_module = create_conv_module(in_channels, out_channels, kernel_size, stride, padding, dilation, groups);
    load_conv_module(conv_module, fp);
    fclose(fp);
    return conv_module;
}

void * test_conv_module(ConvModule * conv_module, float * data,int height,int width){
    int in_size[2] = {height,width};
    int out_size[2];
    float * output = create_conv_output(conv_module, in_size, out_size);
    if(output == NULL){
        perror("create_conv_output");
        exit(-1);
    }
    conv_module_forward(conv_module, data, output,in_size,out_size);
    return output;
}
void print_output(float * output, int height, int width,int channels);
// int run_conv_test(int argc, char * argv[]){
//     char *path = "/home/xs/Code/Python/MachineLearning/base_module/bin/ConvModule.bin";

//     ConvModule * conv_module = build_conv_module(path);
//     ImgData *data = load_img_data("/home/xs/Code/Python/MachineLearning/base_module/data/test.bin");
//     int in_size[2] = {data->height,data->width};
//     int in_channels = data->channels;
//     int out_size[2];
//     // float * input = (float *)malloc(sizeof(float) * in_channels * in_size[0] * in_size[1]);
//     if(in_channels != conv_module->in_channels){
//         printf("error: in_channels != conv_module->in_channels\n");
//         exit(-1);
//     }
//     float * output = create_conv_output(conv_module, in_size, out_size);
//     conv_module_forward(conv_module, data->data, output,in_size,out_size);
//     print_output(output,out_size[0],out_size[1],conv_module->out_channels);

// }
void print_output(float * output, int height, int width,int channels){
    float sum = 0;
    int n = 0;
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            for(int k = 0; k < channels; k++){
                sum += output[i * width * channels + j * channels + k];  
            }
            printf("%3.4f ",sum);
            n++;
            sum = 0;
        }
        printf("\n");
    }
    printf("n = %d h = %d w = %d \n",n,height,width);
}
void print_output_channel(float * output, int height, int width,int channels){
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            for(int k = 0; k < channels; k++){
                printf("%1.1f ",output[i * width * channels + j * channels + k]);  
            }
            printf("\n");
        }
        printf("\n");
    }
}


int run_conv_test(int argc, char * argv[]){
    char *path = "/home/xs/Code/Python/MachineLearning/base_module/bin/ConvModule.bin";

    ConvModule * conv_module = build_conv_module(path);
    float input[9] = {1,2,3,4,5,6,7,8,9};
    int in_size[2] = {3,3};
    int in_channels = 1;
    int out_size[2];

    if(in_channels != conv_module->in_channels){
        printf("error: in_channels != conv_module->in_channels\n");
        exit(-1);
    }
    float * output = create_conv_output(conv_module, in_size, out_size);
    conv_module_forward(conv_module, input, output,in_size,out_size);
    print_output(output,out_size[0],out_size[1],conv_module->out_channels);

}