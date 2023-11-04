import torch
from model.linear import LinearModule

model = LinearModule(5,10)
model.train()
loss = torch.nn.MSELoss()
for i in range(100):
    input = torch.randn(5,5)
    target = torch.concat((input,input * 2),dim=1)
    output = model(input)
    loss_value = loss(output,target)
    loss_value.backward()
    print(loss_value)
    model.zero_grad()
torch.save(model,"pth/LinearModule.pth")

