from test import TestLinearModule
import torch
data = [torch.tensor([[1.0,2,3,100,5]]),
        torch.tensor([[1.0,2,3,863,5]]),
        torch.tensor([[1.0,2,3,4,456]]),
        torch.tensor([[235.0,2,3,4,5]]),
        torch.tensor([[1.0,1285,3,4,5]]),]
test = TestLinearModule(data)
outputs = test.diff()
print()
print(outputs)
for output in outputs:
    for o in output:
        if o > 0.0001:
            print(o)