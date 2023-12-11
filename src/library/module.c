#include "module.h"
#include <stdlib.h>
#include "linear.h"
#include "layernorm.h"
Module * Module_new(void * module,
                    void (*forward)(void * /* module */, Tensor * /* input */,Tensor */*output*/),
                    void (*free_module)(void * /*module*/),
                    Tensor * (*create_output)(void * /* module */ , Tensor * /* input */),
                    module_type type){
    Module * new_module = malloc(sizeof (Module));
    new_module->module = module;
    new_module->forward = forward;
    new_module->free_module = free_module;
    new_module->create_output = create_output;
    new_module->type = type;
    return new_module;
}

Module * linear_module_new (void * module){
    return Module_new(module,(void (*)(void * /* module */, Tensor * /* input */,Tensor */*output*/))forward_linear_module,
                      (void (*)(void * /*module*/))free_linear_module,
                      (Tensor * (*)(void * /* module */ , Tensor * /* input */))create_linear_output,MODULE);
}
Module * layernorm_module_new (void * module){
    return Module_new(module,(void (*)(void * /* module */, Tensor * /* input */,Tensor */*output*/))forward_layernorm_module,
                      (void (*)(void * /*module*/))free_layernorm_module,
                      (Tensor * (*)(void * /* module */ , Tensor * /* input */))create_layernorm_output,MODULE);
}
//Module * FunctionModule_new(Tensor * (*function)(int /*num_params*/,...),int num_params){
//    Module * new_module = malloc(sizeof (Module));
//    new_module->type = FUNCTION;
//    new_module->function = function;
//    new_module->num_params = num_params;
//    return new_module;
//}
Tensor * forward_module(Module * module,Tensor * input){

    Tensor * output = module->create_output(module,input);
    module->forward(module,input,output);
    return output;
}
void free_module(Module * module){
    Module * next = module;
    while (next->type == ABSTRACT_MODULE){
        next = module->module;
        free(module);
        module = next;
    }
    if(module->type == MODULE){
        module->free_module(module->module);
    }
    free(module);
}

ModuleList * ModuleList_new(int base_num){
    ModuleList * module_list = malloc(sizeof(ModuleList));
    module_list->list = malloc(sizeof(Module *) * base_num);
    module_list->outputs = malloc(sizeof(Tensor *) * base_num);
    module_list->num_module = 0;
    module_list->len = base_num;
    return module_list;
}
ModuleList * append_module(ModuleList * module_list,Module * module){
    if(module_list->num_module + 1 >= module_list->len ){
        module_list->len *= 2;
        Module ** new_list = malloc(sizeof(Module *) * module_list->len);
        Tensor ** new_outputs = malloc(sizeof(Tensor *) * module_list->len);
        for(int i = 0;i < module_list->num_module;i++){
            new_list[i] = module_list->list[i];
            new_outputs[i] = module_list->outputs[i];
        }
        free(module_list->list);
        free(module_list->outputs);
        module_list->list = new_list;
        module_list->outputs = new_outputs;
    }
    module_list->list[module_list->num_module++] = module;
    return module_list;
}

Tensor * forward_module_list(ModuleList * module_list, Tensor * input){
    Tensor * output = NULL;
    for (int i = 0; i < module_list->num_module;i++){
        output = forward_module(module_list->list[i],input);
        module_list->outputs[i] = output;
        input = output;
    }
    return output;
}
void freeModuleList(ModuleList *module_list){
    for(int i = 0;i < module_list->num_module;i++){
        free_module(module_list->list[i]);
        free_tensor(module_list->outputs[i]);
    }
    free(module_list->list);
    free(module_list->outputs);
    free(module_list);
}