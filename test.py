from test import TestLinearModule
from test import TestConvModule
# import config._linear as cfg
import config._conv as cfg
test_Modules = dict(
    LinearModule=TestLinearModule,
    ConvModule =TestConvModule,

)
test = test_Modules[cfg.config['name']](cfg.config['data'])
outputs = test.diff()
print()

import numpy as np
for output in outputs:
    output = np.array(output)
    indices = np.where(output > 0.0001)
    values = output[indices]
    print("Indices:", indices)
    print("Values:", values)


         