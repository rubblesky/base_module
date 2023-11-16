import numpy as np
from test import TestLinearModule
from test import TestConvModule
from test import TestBatchNormModule
from test import TestLayerNormModule

from import_config import configs


test_Modules = dict(
    LinearModule=TestLinearModule,
    ConvModule=TestConvModule,
    BatchNormModule=TestBatchNormModule,
    LayerNormModule=TestLayerNormModule,
)
total_num = 0
for i, cfg in enumerate(configs):

    print("\033[1;31;40mTest \033[0m", i, ": " + cfg['config']['name'])
    test = test_Modules[cfg['config']['name']](data=cfg['test_data'], model_path=cfg['config']['model_path'],
                                               bin_path=cfg['config']['bin_path'], )
    outputs = test.diff()

    for output in outputs:
        output = np.array(output)
        indices = np.where(output > 0.0001)
        values = output[indices]
        print("\033[1;33;40m Indices: \033[0m", indices)
        print("\033[1;33;40m Values: \033[0m", values)
        total_num += len(values)
print("total failed: ", total_num)
