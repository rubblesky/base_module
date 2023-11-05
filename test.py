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
# print(outputs)
# for output in outputs:
#     for o in output:
#         if o > 0.0001:
#             print(o)            