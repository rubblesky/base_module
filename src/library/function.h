#ifndef FUNCTION_H
#define FUNCTION_H
#include "tensor.h"
#include <stdarg.h>

Tensor * permute_function(int num_params, ...){
    va_list args;
    va_start(args, num_params);
    return permute(va_arg(args, Tensor*),va_arg(args, int*),va_arg(args, int));
}


#endif
