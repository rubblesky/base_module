#ifndef CLIP_VISION_H
#define CLIP_VISION_H
#include "../library/module.h"
typedef Tensor PositionalEmbedding;
typedef Tensor ClassEmbedding;
typedef ModuleList Transformer;

typedef struct VisionEncoder{
    ClassEmbedding * class_embedding;
    PositionalEmbedding * positional_embedding;
    Transformer * transformer;
    Module * conv1;
    Module * ln_pre;
    Module * proj;

}VisionEncoder;

typedef struct VisionTransformerLayer{
    int num_head;
    Module * in_proj;
    Module * out_proj;
    Module * ln1;
    Module * mlp_fc;
    Module * mlp_proj;
    Module * ln2;

    Tensor * in_proj_output;
    Tensor * out_proj_output;
    Tensor * ln1_output;
    Tensor * mlp_fc_output;
    Tensor * mlp_proj_output;
    Tensor * ln2_output;

} VisionTransformerLayer;

VisionTransformerLayer * create_vision_transformer_layer(int in_channel,int out_channel,int d_model,int num_head);
void free_transformer_layer(VisionTransformerLayer * vlt);
void forward_transformer_layer(VisionTransformerLayer * module,Tensor * input,Tensor * output);
Tensor * create_transformer_layer_output(VisionTransformerLayer * module, Tensor * input);
void free_transformer_layer_output(Tensor * output);

int run_transformer_layer_test(int argc, char * argv[]);
void forward_vision_module(VisionEncoder * module,Tensor * input, Tensor * output);
#endif //CLIP_VISION_H
