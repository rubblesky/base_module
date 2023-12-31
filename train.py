import torch
import torch.optim as optim
from model import *

from import_config import configs, type

if type == 'module':
    for cfg in configs:

        model = cfg['config']['module'](**cfg['config']['params'])
        model.train()
        loss = cfg['config']['loss']
        if 'train_data' in cfg:
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
            print("training  " + cfg['config']['name'])
            for input,target in cfg['train_data']:
                output = model(input)
                loss_value = loss(output,target)
                loss_value.backward()
                optimizer.step()
                # print(loss_value)
                model.zero_grad()
        torch.save(model,cfg['config']['model_path'])

