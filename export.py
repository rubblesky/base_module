from model.linear import LinearModule
import torch
import struct

def serialize_fp32(file, tensor):
    """ writes one fp32 tensor to file that is open in wb mode """
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    b = struct.pack(f'{len(d)}f', *d)
    file.write(b)


model = torch.load("./pth/LinearModule.pth")

with open("./bin/LinearModule.bin", "wb") as f:
    w = model.linear.weight.view(-1)
    b = model.linear.bias
    serialize_fp32(f, w)    
    serialize_fp32(f, b)
print("finished")

