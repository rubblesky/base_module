#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>
Tensor * Tensor_new(int num_dim,int shape[],float *data){
    if(num_dim > MAX_DIM){
        printf("Error : dim must less than MAX_DIM, get %d\n",num_dim);
        exit(-1);
    }
    Tensor * tensor =  malloc(sizeof(Tensor));
    tensor->num_dim = num_dim;
    tensor->num_features = 1;
    for(int i = 0;i < num_dim;i++){
        tensor->shape[i] = shape[i];
        tensor->num_features *= shape[i];
    }
    tensor->data = malloc(sizeof(float) * tensor->num_features);
    for(int i = 0;i < tensor->num_features;i++){
        tensor->data[i] = data[i];
    }
    return tensor;
}
Tensor * Tensor_copy(Tensor * tensor){
    Tensor * new_tensor =  malloc(sizeof(Tensor));
    new_tensor->num_dim = tensor->num_dim;
    new_tensor->num_features = tensor->num_features;
    for(int i = 0;i < tensor->num_dim;i++){
        new_tensor->shape[i] = tensor->shape[i];
    }
    new_tensor->data = malloc(sizeof(float) * tensor->num_features);
    for(int i = 0;i < tensor->num_features;i++){
        new_tensor->data[i] = tensor->data[i];
    }
    return new_tensor;
}
Tensor * Tensor_like(Tensor * tensor){
    Tensor * new_tensor =  malloc(sizeof(Tensor));
    new_tensor->num_dim = tensor->num_dim;
    new_tensor->num_features = tensor->num_features;
    for(int i = 0;i < tensor->num_dim;i++){
        new_tensor->shape[i] = tensor->shape[i];
    }
    new_tensor->data = malloc(sizeof(float) * tensor->num_features);
    return new_tensor;
}
Tensor * Tensor_init(int num_dim,int shape[]){
    if(num_dim > MAX_DIM){
        printf("Error : dim must less than MAX_DIM, get %d\n",num_dim);
        exit(-1);
    }
    Tensor * tensor =  malloc(sizeof(Tensor));
    tensor->num_dim = num_dim;
    tensor->num_features = 1;
    for(int i = 0;i < num_dim;i++){
        tensor->shape[i] = shape[i];
        tensor->num_features *= shape[i];
    }
    tensor->data = calloc(tensor->num_features,sizeof(float));
    return tensor;
}
void free_tensor(Tensor * tensor){
    free(tensor->data);
    free(tensor);
}
Tensor * reshape_(Tensor * tensor,int num_dim,int *shape){
    //assert all memory are congious
    if(num_dim > MAX_DIM){
        printf("Error : dim must less than MAX_DIM, get %d\n",num_dim);
        exit(-1);
    }
    tensor->num_dim = num_dim;
    for(int i = 0;i < num_dim;i++){
        tensor->shape[i] = shape[i];
    }
    return tensor;
}
Tensor * permute(Tensor *tensor,int * dims, int num_dim){
    if (num_dim != tensor->num_dim){
        printf("Error: num_dim in transpose must be consistent with tensor->num_dim, get %d but tensor->num_dim = %d\n",num_dim,tensor->num_dim);
        exit(-1);
    }
    int new_shape[MAX_DIM];
    for(int i = 0;i < num_dim;i++){
        new_shape[i] = tensor->shape[dims[i]];
    }    
    Tensor * transposed_tensor = Tensor_init(num_dim,new_shape);
    int strides[MAX_DIM + 1];
    int transposed_strides[MAX_DIM + 1];
    for (int i = 0;i <= MAX_DIM;i++){
        strides[i] = 1;
        transposed_strides[i] = 1;
    }
    int last = 1;
    int transposed_last = 1;
    for(int i = num_dim - 1;i >=0; i--){
        strides[i] = last * strides[i + 1];
        last = tensor->shape[i];
        transposed_strides[i] = transposed_last * transposed_strides[i + 1];
        transposed_last = transposed_tensor->shape[i];
    }
    float * transposed_data = transposed_tensor -> data;
    int tensor_indices[MAX_DIM] = {0};
    int p;
    for(int i = 0;i < tensor->num_features;i++){
        int s = i;
        p = 0;
        //TODO: change div to add
        for(int n = 0;n < num_dim;n++){
            s = s / strides[n];
            tensor_indices[n] = s;
            s = i % strides[n];
        }
        for(int n = 0;n < num_dim;n++){
            p += tensor_indices[dims[n]] * transposed_strides[n];
        }
        transposed_data[p] = tensor->data[i];
    }
    return transposed_tensor;
}

Tensor * relu(Tensor * input){
    Tensor * output = Tensor_copy(input);
    for (int i = 0; i < output->num_features;i++){
        if (output->data[i] < 0){
            output->data[i] = 0;
        }
    }
    return output;
}

void print_tensor(Tensor* tensor) {
    int indices[MAX_DIM] = {0};
    for (int i = 0; i < tensor->num_features; ++i) {
        for (int dim = tensor->num_dim - 1; dim >= 0; --dim) {
            printf("%d ", indices[dim]);
        }
        printf(": %f\n", tensor->data[i]);

        ++indices[0];
        for (int dim = 0; dim < tensor->num_dim - 1; ++dim) {
            if (indices[dim] >= tensor->shape[tensor->num_dim - 1 - dim]) {
                indices[dim] = 0;
                ++indices[dim + 1];
            }
        }
    }
}

int run_permute_test(int argc, char * argv[]){
    float input_data[18] = {0,1,2,3,4,5,6,7,8, 9, 10, 11, 12};
    int input_shape[3] = {2,2,3};
    int dims[3] = {1,2,0};
    Tensor * input = Tensor_new(3,input_shape,input_data);
    Tensor * output = permute(input,dims,3);
    print_tensor(input);
    printf("\n");
    print_tensor(output);
    free_tensor(input);
    free_tensor(output);
}