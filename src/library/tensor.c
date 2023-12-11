#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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
Tensor * cat(Tensor * t1,Tensor * t2,int dim){
    if(t1->num_dim != t2->num_dim){
        return NULL;
    }
    for(int i = 0;i < t1->num_dim;i++){
        if(i == dim || t1->shape[i] != t2->shape[i]){
            return NULL;
        }
    }
    int num_dim = t1->num_dim;
    int shape[num_dim];
    for(int i = 0;i < num_dim;i++){
        shape[i] = t1->shape[i];
    }
    shape[dim] = t1->shape[dim] + t2->shape[dim];
    Tensor * result = Tensor_init(num_dim,shape);
    int index[num_dim];
    for (int i = 0;i < num_dim;i++){
        index[i] = 0;

    }
    for(int i = 0;i < t1->num_features + t2->num_features;i++){
        index[num_dim - 1]++;
        for(int j = num_dim - 1;index[j] >= shape[j] && j > 0;j--){
            index[j] = 0;
            index[j - 1]++;
        }
        int k = index[0];
        for (int j = 1;j < num_dim;j++){
            k *= shape[j];
            k += index[j];
        }
        float * source = NULL;
        if (index[dim] > t1->shape[dim]){
            int s = index[0];
            for (int j = 1;j < num_dim;j++){
                s *= t1->shape[j];
                s += index[j];
            }
            source = t1->data + s;
        }else{
            index[dim] -= t1->shape[dim];
            int s = index[0];
            for (int j = 1;j < num_dim;j++){
                s *= t2->shape[j];
                s += index[j];
            }
            source = t2->data + s;
        }
        result->data[k] = *source;
    }
    return result;
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
Tensor * sigmoid(Tensor * input){
    Tensor * output = Tensor_copy(input);
    for (int i = 0; i < output->num_features;i++){
        float x = input->data[i];
        output->data[i] = 1.f / (1.f + expf(-x));

    }
    return output;
}
Tensor * quick_gelu(Tensor * input){
    Tensor * output = Tensor_copy(input);
    for (int i = 0; i < output->num_features;i++){
        float x = input->data[i];
        output->data[i] = x *  1.f / (1.f + expf(-1.702 * x));
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


TensorList * TensorList_new(int base_num){
    TensorList * tensor_list = malloc(sizeof(TensorList));
    tensor_list->list = calloc(base_num, sizeof(Tensor *));
    tensor_list->len = base_num;
    tensor_list->num_tensor = 0;
    return tensor_list;
}
TensorList * append_tensor(TensorList * tensor_list,Tensor * tensor){
    if(tensor_list->num_tensor + 1 >= tensor_list->len ){
        tensor_list->len *= 2;
        Tensor ** new_list = malloc(sizeof(Tensor *) * tensor_list->len);
        for(int i = 0;i < tensor_list->num_tensor;i++){
            new_list[i] = tensor_list->list[i];
        }
        free(tensor_list->list);
        tensor_list->list = new_list;
    }
    tensor_list->list[tensor_list->num_tensor++] = tensor;
    return tensor_list;
}
void freeTensorList(TensorList * tensor_list){
    for(int i = 0;i < tensor_list->num_tensor;i++){
        free_tensor(tensor_list->list[i]);
    }
    free(tensor_list->list);
    free(tensor_list);
}
Tensor * get_tensor(TensorList * tensor_list,int id){
    if(id < 0 || id > tensor_list->num_tensor){
        return NULL;
    }
    return tensor_list->list[id];
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