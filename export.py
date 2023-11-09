
import torch
import struct
from config._conv import configs


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
    bw = model.bn.weight.view(-1)
    bb = model.bn.bias
    serialize_fp32(file, cw)    
    serialize_fp32(file, cb)
    serialize_fp32(file, bw)    
    serialize_fp32(file, bb)

functions = dict(
    LinearModule = export_linear,
    ConvModule = export_conv
    
)
if __name__ == "__main__":
    for cfg in configs:
        export(cfg['config']['model_path'],cfg['config']['bin_path'],functions[cfg['config']['name']])