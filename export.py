
import torch
import struct
from config._batchnorm import configs


def serialize_fp32(file, tensor):
    """ writes one fp32 tensor to file that is open in wb mode """
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    b = struct.pack(f'{len(d)}f', *d)
    file.write(b)


def export(model_path,bin_path,export_func):

    model = torch.load(model_path)

    with open(bin_path, "wb") as f:
        export_func(model,f)
    print(model_path + " export successfully")


def export_linear(model,file):
    w = model.linear.weight.view(-1)
    b = model.linear.bias
    serialize_fp32(file, w)    
    serialize_fp32(file, b)

def export_conv(model,file):
    in_channels = model.conv.in_channels
    out_channels = model.conv.out_channels
    kernel_size = model.conv.kernel_size if len(model.conv.kernel_size)==2 else (model.conv.kernel_size,model.conv.kernel_size)
    stride = model.conv.stride if len(model.conv.stride)==2 else (model.conv.stride,model.conv.stride)
    padding = model.conv.padding if len(model.conv.padding)==2 else (model.conv.padding,model.conv.padding)
    dilation = model.conv.dilation if len(model.conv.dilation)==2 else (model.conv.dilation,model.conv.dilation)
    groups = model.conv.groups
    header = struct.pack(f'{11}i',in_channels,out_channels,kernel_size[0],kernel_size[1],stride[0],stride[1],padding[0],padding[1],dilation[0],dilation[1],groups)
    file.write(header)
    cw = model.conv.weight.view(-1)
    cb = model.conv.bias

    serialize_fp32(file, cw)    
    serialize_fp32(file, cb)

def export_batchnorm(model,file):
    num_features = model.bn.num_features
    eps = model.bn.eps
    header = struct.pack(f'if',num_features,eps)
    file.write(header)
    w = model.bn.weight
    b = model.bn.bias
    m = model.bn.running_mean
    v = model.bn.running_var
    serialize_fp32(file,w)
    serialize_fp32(file,b)
    serialize_fp32(file,m)
    serialize_fp32(file,v)

def export_layernorm(model,file):
    # LayerNorm
    normalized_shape = model.ln.normalized_shape
    eps = model.ln.eps
    header = struct.pack(f'if',normalized_shape,eps)
    file.write(header)
    w = model.ln.weight
    b = model.ln.bias
    serialize_fp32(file,w)
    serialize_fp32(file,b)


functions = dict(
    LinearModule = export_linear,
    ConvModule = export_conv,
    BatchNormModule = export_batchnorm,
    LayerNormModule = export_layernorm,
    
)
if __name__ == "__main__":
    for cfg in configs:
        export(cfg['config']['model_path'],cfg['config']['bin_path'],functions[cfg['config']['name']])