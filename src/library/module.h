//
// Created by xs on 11/21/23.
//

#ifndef MODULE_H
#define MODULE_H
#include "tensor.h"
typedef enum module_type{MODULE,FUNCTION,ABSTRACT_MODULE} module_type;
typedef struct Module{
    void * module;
    void (*forward)(void * /* module */, Tensor * /* input */,Tensor */*output*/);
//    Tensor * (*function)(int /*num_params*/,...);
    void (*free_module)(void * /*module*/);
    Tensor * (*create_output)(void * /* module */ , Tensor * /* input */);
    module_type type;
//    int num_params;
}Module;

Module * Module_new(void * module,
                    void (*forward)(void * /* module */, Tensor * /* input */,Tensor */*output*/),
                    void (*free_module)(void * /*module*/),
                    Tensor * (*create_output)(void * /* module */ , Tensor * /* input */),
                    module_type type);
Module * linear_module_new (void * module);
Module * layernorm_module_new (void * module);
//Module * FunctionModule_new(Tensor * (*function)(int /*num_params*/,...),int num_params);
Tensor * forward_module(Module * module,Tensor * input);
void free_module(Module * module);

typedef struct ModuleList{
    Module ** list;
    Tensor ** outputs;
    int num_module;
    int len;
}ModuleList;

ModuleList * ModuleList_new(int base_num);
ModuleList * append_module(ModuleList * module_list,Module * module);
Tensor * forward_module_list(ModuleList * module_list, Tensor * input);
void freeModuleList(ModuleList *module_list);

#endif
