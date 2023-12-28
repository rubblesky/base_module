#ifndef TENSOR_H
#define TENSOR_H
#define MAX_DIM 10
typedef struct Tensor
{
    int num_dim;
    int shape[MAX_DIM];
    int num_features;
    float* data;
} Tensor;
Tensor * Tensor_new(int num_dim,int shape[],float *data);
Tensor * Tensor_copy(Tensor * tensor);
Tensor * Tensor_like(Tensor * tensor);
Tensor * Tensor_init(int num_dim,int shape[]);
void free_tensor(Tensor * tensor);
Tensor * reshape_(Tensor * tensor,int num_dim,int *shape);
Tensor * permute(Tensor *tensor,int * dims, int num_dim);
Tensor * cat(Tensor * t1,Tensor * t2,int dim);
Tensor * add(Tensor * t1,Tensor * t2);
Tensor * slice(Tensor * t,int start[],int end[],int dim);

Tensor * relu(Tensor * input);

void print_tensor(Tensor * tensor);

typedef struct TensorList{
    int num_tensor;
    int len;
    Tensor ** list;
}TensorList;
TensorList * TensorList_new(int base_num);
TensorList * append_tensor(TensorList * tensor_list,Tensor * tensor);
void freeTensorList(TensorList * tensor_list);
Tensor * get_tensor(TensorList * tensor_list,int id);
#endif