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
void reshape(Tensor * tensor,int num_dim,int *shape);
Tensor * transpose(Tensor *tensor,int * dims, int num_dim);

void print_tensor(Tensor * tensor);
#endif