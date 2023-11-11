#ifndef OP_H
#define OP_H
void matmul(float * output, float * x, float * weight, int m, int n);
void add(float *output,float * input, int n);
void relu(float * output,float * input,int n);
#endif