#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>
Tensor * Tensor(int num_dim,int shape[],float *data){
    if(num_dim > MAX_DIM){
        printf("Error : dim must less than MAX_DIM, get %d\n",num_dim);
        exit(-1);
    }
    Tensor * tensor =  malloc(sizeof(Tensor));
    tensor->num_dim = num_dim;
    tensor->num_feature = 1;
    for(int i = 0;i < num_dim;i++){
        tensor->shape[i] = shape[i];
        tensor->num_feature *= shape[i];
    }
    tensor->data = malloc(sizeof(float) * tensor->num_feature);
    for(int i = 0;i < tensor->num_feature;i++){
        tensor->data[i] = data[i];
    }
    return tensor;
}
Tensor * Tensor_copy(Tensor * tensor){
    Tensor * new_tensor =  malloc(sizeof(Tensor));
    new_tensor->num_dim = tensor->num_dim;
    new_tensor->num_feature = tensor->num_feature;
    for(int i = 0;i < num_dim;i++){
        new_tensor->shape[i] = tensor->shape[i];
    }
    new_tensor->data = malloc(sizeof(float) * tensor->num_feature);
    for(int i = 0;i < tensor->num_feature;i++){
        new_tensor->data[i] = tensor->data[i];
    }
    return new_tensor;
}
Tensor * Tensor(int num_dim,int shape[]){
    if(num_dim > MAX_DIM){
        printf("Error : dim must less than MAX_DIM, get %d\n",num_dim);
        exit(-1);
    }
    Tensor * tensor =  malloc(sizeof(Tensor));
    tensor->num_dim = num_dim;
    tensor->num_feature = 1;
    for(int i = 0;i < num_dim;i++){
        tensor->shape[i] = shape[i];
        tensor->num_feature *= shape[i];
    }
    tensor->data = calloc(tensor->num_feature,sizeof(float));
    return tensor;
}
void free_tensor(Tensor * tensor){
    free(tensor->data);
    free(tensor);
}
void reshape(Tensor * tensor,int num_dim.int *shape){
    //assert all memory are congious
    if(num_dim > MAX_DIM){
        printf("Error : dim must less than MAX_DIM, get %d\n",num_dim);
        exit(-1);
    }
    tensor->dim = num_dim;
    for(int i = 0;i < num_dim;i++){
        tensor->shape[i] = shape[i];
    }
}
Tensor * transpose(Tensor *tensor,int * dims, int num_dim){
    if (num_dim != tensor->num_dim){
        printf("Error: num_dim in transpose must be consistent with tensor->num_dim, get %d but tensor->num_dim = %d\n",num_dim,tensor->num_dim);
        exit(-1);
    }
    int new_shape[10];
    for(int i = 0;i < num_dim;i++){
        new_shape[i] = tensor->shape[dims[i]];
    }    
    Tensor * transposed_tensor = Tensor_init(num_dim,new_shape);
    int strides[MAX_DIM + 1];
    for (int i = 0;i <= MAX_DIM;i++){
        strides[i] = 1;
    }
    int last = 1;
    for(int i = num_dim - 1;i >=0; i--){
        strides[i] = last * strides[i + 1];
        last = tensor->shape[i];
    }
    float * transposed_data = transposed_tensor -> data;
    int s,p;
    for(int i = 0;i < tensor->num_feature;i++){
        s = i;
        p = 0;
        for(int n = 0;n < dim;n++){
            s = s / stride[n];
            p += s * stride[dims[n]];
            s = s % stride[n];
        }
        transposed_data[p] = tensor->data[i];
    }
    return transposed_tensor;
}