import torch
from model import *
from config._batchnorm import configs
for cfg in configs:
    model = cfg['config']['module'](**cfg['config']['params'])
    model.train()
    loss = cfg['config']['loss']
    # for i in range(100):
    #     input = torch.randn(5,5)
    #     target = torch.concat((input,input * 2),dim=1)
    #     output = model(input)
    #     loss_value = loss(output,target)
    #     loss_value.backward()
    #     print(loss_value)
    #     model.zero_grad()
    torch.save(model,cfg['config']['model_path'])

