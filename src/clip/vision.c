#include "vision.h"
#include "../library/layernorm.h"
#include "../library/linear.h"
#include <stdlib.h>
VisionTransformerLayer * create_transformer_layer(int in_channel,int out_channel,int d_model,int num_head){
    VisionTransformerLayer * vtl = malloc(sizeof(VisionTransformerLayer));
    LinearModule * in_proj = create_linear_module(in_channel,out_channel);
    LinearModule * out_proj = create_linear_module(out_channel,out_channel);
    int shape[1];
    shape[0] = out_channel;
    LayerNormModule * ln1 = create_layernorm_module(1,shape,1e-6);
    LinearModule * mlp_fc = create_linear_module(out_channel,d_model);
    LinearModule * mlp_proj = create_linear_module(d_model,out_channel);
    LayerNormModule * ln2 = create_layernorm_module(1,shape,1e-6);

    vtl->num_head = num_head;
    vtl->in_proj = linear_module_new(in_proj);
    vtl->out_proj = linear_module_new(out_proj);

    vtl->ln1 = layernorm_module_new(ln1);

    vtl->mlp_fc = linear_module_new(mlp_fc);
    vtl->mlp_proj = linear_module_new(mlp_proj);

    vtl->ln2 = layernorm_module_new(ln2);

    vtl->in_proj_output = NULL;
    vtl->out_proj_output = NULL;
    vtl->ln1_output = NULL;
    vtl->mlp_fc_output = NULL;
    vtl->mlp_proj_output = NULL;
    vtl->ln2_output = NULL;

    return vtl;
}

void free_transformer_layer(VisionTransformerLayer * vlt){
    free_module(vlt->in_proj);
    free_module(vlt->out_proj);
    free_module(vlt->ln1);
    free_module(vlt->mlp_fc);
    free_module(vlt->mlp_proj);
    free_module(vlt->ln2);

    free_tensor(vlt->in_proj_output);
    free_tensor(vlt->out_proj_output);
    free_tensor(vlt->ln1_output);
    free_tensor(vlt->mlp_fc_output);
    free_tensor(vlt->mlp_proj_output);
    free_tensor(vlt->ln2_output);

    free(vlt);
}

void forward_transformer_layer(VisionTransformerLayer *vtl,Tensor * input,Tensor * output){

    int bsz = input->shape[0];
    int len_q = input->shape[1];
    int len_kv = input->shape[1];
    int embed_dim = input->shape[2];
    int head_dim = input->shape[2] / vtl->num_head;
    //assert (input->shape[2] % vlt->num_head) == 0
    if(vtl->ln1_output == NULL){
        vtl->ln1_output = vtl->ln1->create_output(vtl->ln1,input);
    }
    if(vtl->in_proj_output == NULL){
        vtl->in_proj_output = vtl->in_proj->create_output(vtl->in_proj,vtl->ln1_output);
    }




    vtl->ln1->forward(vtl->ln1,input,vtl->ln1_output);
    vtl->in_proj->forward(vtl->in_proj,vtl->ln1_output,vtl->in_proj_output);
    int shape[MAX_DIM] = {bsz,len_q,3,embed_dim};
    vtl->in_proj_output = reshape_(vtl->in_proj_output,4,shape);


}

void forward_vision_module(VisionEncoder * module,Tensor * input, Tensor * output){
    Tensor * conv1_output = module->conv1->create_output(module->conv1,input);
    module->conv1->forward(module->conv1,input,conv1_output);

    int reshape_size[3];
    reshape_size[0] = input->shape[0];
    reshape_size[1] = input->shape[1];
    reshape_size[2] = input->shape[2] * input->shape[3];
    conv1_output = reshape_(conv1_output,3,reshape_size);
    int permute_dims[3] = {1,0,2};
    Tensor * conv1_output_permute = permute(conv1_output,permute_dims,3); // new
    int zero_shape[3] = {conv1_output_permute->shape[0],1,conv1_output_permute->shape[conv1_output_permute->num_dim-1]};
    Tensor * zero = Tensor_init(3,zero_shape); //new
    Tensor * class_embedding = add(module->class_embedding,zero);   //new
    Tensor * cat_output = cat(class_embedding,conv1_output_permute,1); // new
    Tensor * embedding = add(module->positional_embedding,cat_output); //new
    Tensor * in_pre_output = module->ln_pre->create_output(module->ln_pre,embedding); //new
    module->ln_pre->forward(module->ln_pre,embedding,in_pre_output);
    //Tensor *  embedding_permute = permute(embedding,permute_dims,3); //new we do not need it because we choose batch_first=True

    module->transformer; //TODO







}