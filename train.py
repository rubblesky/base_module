import torch
import torch.optim as optim
from model import *
from config._linear import configs
for cfg in configs:
    
    model = cfg['config']['module'](**cfg['config']['params'])
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    model.train()
    loss = cfg['config']['loss']
    if 'train_data' in cfg:
        print("training  " + cfg['config']['name'])
        for input,target in cfg['train_data']:
            output = model(input)
            loss_value = loss(output,target)
            loss_value.backward()
            optimizer.step()
            # print(loss_value)
            model.zero_grad()
    torch.save(model,cfg['config']['model_path'])

