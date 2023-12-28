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
Tensor * squeeze_(Tensor * tensor, int n){
    if(n < tensor->num_dim && tensor->shape[n] == 1){
        int new_shape[MAX_DIM];
        int *ns = new_shape;
        for(int i = 0;i < tensor->num_dim;i++){
            if(i == n){
                continue;
            } else{
                *ns = tensor->shape[i];
            }
        }
        return  reshape_(tensor,tensor->num_dim - 1,new_shape);
    } else{
        return NULL;
    }
}
Tensor * unsqueeze_(Tensor * tensor, int n){
    if(n <= tensor->num_dim  && n >= 0){
        int new_shape[MAX_DIM];
        int *ns = new_shape;
        for(int i = 0;i < tensor->num_dim;i++){
            if(i == n){
                *ns = 1;
            } else{
                *ns = tensor->shape[i];
            }
        }
        return  reshape_(tensor,tensor->num_dim + 1,new_shape);
    } else{
        return NULL;
    }
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
        if(i != dim && t1->shape[i] != t2->shape[i]){
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
    for(int i = 0;i < t1->num_features + t2->num_features;i++,index[num_dim - 1]++){

        for(int j = num_dim - 1;index[j] >= shape[j] && j > 0;j--){
            index[j] = 0;
            index[j - 1]++;
        }

        float * source = NULL;
        if (index[dim] < t1->shape[dim]){
            int s = index[0];
            for (int j = 1;j < num_dim;j++){
                s *= t1->shape[j];
                s += index[j];
            }
            source = t1->data + s;
        }else{
            int s = index[0];
            for (int j = 1;j < num_dim;j++){
                s *= t2->shape[j];
                if(j == dim){
                    s += index[j] - t1->shape[j];
                }else{
                    s += index[j];
                }
                
            }
            source = t2->data + s;
        }

        result->data[i] = *source;
    }
    return result;
}
Tensor * add(Tensor * t1,Tensor * t2){
    if(t1->num_dim < t2->num_features){
        // ensure t1->num_dim >= t2->num_dim
        Tensor * tmp = t1;
        t1 = t2;
        t2 = tmp;
    }
    int t2_shape[MAX_DIM] = {0};
    int *sp2 = t2_shape;
    for(int i = 0;i < t1->num_dim-t2->num_dim;i++){
        *sp2++ = 1;
    }
    for(int i = 0;i < t2->num_dim;i++){
        sp2[i] = t2->shape[i];
    }
    t2 = reshape_(t2,t1->num_dim,t2_shape);
    int shape[MAX_DIM];
    int *sp = shape;
    int *sp1 = t1->shape;
    sp2 = t2->shape;
    for(int i = 0;i < t1->num_dim;sp2++,sp1++,sp++,i++){
        if(*sp1 != *sp2 && *sp1 != 1 && *sp2 != 1){
            return NULL;
        }
        *sp = (*sp1 > *sp2)?*sp1:*sp2;
    }
    Tensor *result = Tensor_init(t1->num_dim,shape);
    int index[MAX_DIM] = {0};
    for(int i = 0;i < result->num_features;i++,index[result->num_dim - 1]++) {
        for (int j = result->num_dim - 1; index[j] >= shape[j] && j > 0; j--) {
            index[j] = 0;
            index[j - 1]++;
        }

        int s1 = 1==t1->shape[0]?0:index[0];
        for (int j = 1;j < t1->num_dim;j++){
            s1 *= t1->shape[j];
            s1 += 1==t1->shape[j]?0:index[j];
        }
        int s2 = 1==t2->shape[0]?0:index[0];
        for (int j = 1;j < t2->num_dim;j++){
            s2 *= t2->shape[j];
            s2 += 1==t2->shape[j]?0:index[j];
        }
        result->data[i] = t1->data[s1] + t2->data[s2];

    }
    return result;
}
Tensor * matmul(Tensor * t1, Tensor * t2){
    int n1 = t1->num_dim - 1;
    int d1 = t1->shape[n1];
    int n2 = t2->num_dim <= 2 ? 0:t2->num_dim - 2;
    int d2 = t2->shape[n2];
    if(d1 != d2){return NULL;}
    else{
        int num_dim = t1->num_dim + t2->num_dim -2;
        int shape[MAX_DIM];
        int *s = shape;
        for(int i = 0; i < n1;i++,s++){
            *s = t1->shape[i];
        }
        for(int i = n1 + 1;i < t2->num_dim;i++,s++){
            *s = t2->shape[i];
        }
        Tensor * result = Tensor_init(num_dim,shape);

        int index[MAX_DIM] = {0};
        for(int i = 0;i < result->num_features;i++,index[result->num_dim - 1]++) {
            for (int j = result->num_dim - 1; index[j] >= shape[j] && j > 0; j--) {
                index[j] = 0;
                index[j - 1]++;
            }

            for(int k = 0;k < d1;k++){
                int m1 = 0;
                for(int l1 = 0;l1 < t1->num_dim -1;l1++){
                    m1 += index[l1] * t1->shape[l1];
                }
                int m2 = 0;
                for(int l2 = 0;l2 < t2->num_dim ;l2++){
                    if(l2 != n2){
                        m2 += index[l2 + t1->num_dim - 1] * t1->shape[l2];
                    }else{
                        m2 += k * t1->shape[l2];
                    }

                }
                result->data[i] += t1->data[m1 + k] * t2->data[m2];
            }
        }

        return result;
    }
}
Tensor * slice(Tensor * t,int start[],int end[],int dim){
    if(dim!=t->num_dim){
        return NULL;
    }
    int shape[MAX_DIM] = {0};
    int num_features = 1;
    for(int i = 0;i < dim ;i++){
        shape[i] = end[i] - start[i];
        num_features *= shape[i];
    }
    Tensor * result = Tensor_init(dim,shape);
    int index[MAX_DIM] = {0};
    int origin_index[MAX_DIM];
    for(int i = 0;i < dim;i++){
        origin_index[i] = start[i];
    }

    for(int i = 0;i < num_features;i++,index[dim-1]++,origin_index[dim-1]++){
        for(int j = dim - 1;index[j] >= shape[j] && j > 0;j--){
            index[j - 1]++;
            index[j] = 0;
            origin_index[j - 1] = index[j - 1] + start[j - 1];
            origin_index[j] = index[j] + start[j];
        }
        int origin = 0;
        for(int k = 0;k < dim;k++){
            origin *= t->shape[k];
            origin += origin_index[k];
        }
        result->data[i] = t->data[origin];
    }
    return result;
}
//int * tensor_iterators(Tensor * t,int i,int index[]){
//    for(int j = t->num_dim - 1;index[j] >= t->shape[j] && j > 0;j--){
//        index[j] = 0;
//        index[j - 1]++;
//    }
//}
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
int run_cat_test(int argc, char * argv[]){
    float input_data[18] = {0,1,2,3,4,5,6,7,8, 9, 10, 11, 12};
    int input_shape[3] = {2,2,3};
    Tensor * input1 = Tensor_new(3,input_shape,input_data);
    Tensor * input2 = Tensor_new(3,input_shape,input_data);
    Tensor * output = cat(input1,input2,1);

    print_tensor(input2);
    printf("\n");
    print_tensor(output);
    free_tensor(input1);
    free_tensor(input2);
    free_tensor(output);
}
int run_add_test(int argc, char * argv[]){
    float input_data[18] = {0,1,2,3,4,5,6,7,8, 9, 10, 11, 12};
    int input_shape1[3] = {2,1,3};
    int input_shape2[3] = {1,2,3};
    Tensor * input1 = Tensor_new(3,input_shape1,input_data);
    Tensor * input2 = Tensor_new(3,input_shape2,input_data);
    Tensor * output = add(input1,input2);

    print_tensor(input2);
    printf("\n");
    print_tensor(output);
    free_tensor(input1);
    free_tensor(input2);
    free_tensor(output);
}
int run_slice_test(int argc, char * argv[]){
    float input_data[18] = {0,1,2,3,4,5,6,7,8, 9, 10, 11, 12};
    int input_shape[3] = {2,2,3};
    int start[3] = {0,0,0};
    int end[3] = {1,2,3};
    Tensor * input = Tensor_new(3,input_shape,input_data);
    Tensor * output = slice(input,start,end,3);
    print_tensor(input);
    printf("\n");
    print_tensor(output);
    free_tensor(input);
    free_tensor(output);
}