import numpy as np
from test import TestModule, TestFunction
from import_config import configs,type

total_num = 0
for i, cfg in enumerate(configs):

    print("\033[1;31;40mTest \033[0m", i, ": " + cfg['config']['name'])
    if type == 'module':
        test = TestModule(cfg)
    elif type == 'function':
        test = TestFunction(cfg)
    outputs = test.diff()

    for output in outputs:
        output = np.array(output)
        indices = np.where(output > 0.0001)
        values = output[indices]
        print("\033[1;33;40m Indices: \033[0m", indices)
        print("\033[1;33;40m Values: \033[0m", values)
        total_num += len(values)
print("total failed: ", total_num)
