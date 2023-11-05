
class TestModule(object):
    """
    A test module to test the Ansible test runner.
    """
    def __init__(self, data,c_data,module,c_module, c_test_func):
        self.module = module
        self.data = data
        self.c_module = c_module
        self.c_test_func = c_test_func
        self.c_data = c_data
    def tests(self):
        outputs = []
        for test in self.data:
            outputs.append(self.module(test))
        return outputs
    
    def c_tests(self):
        outputs = []
        for test in self.c_data:
            output = self.c_test_func(self.c_module,*test)
            outputs.append(output)
        return outputs
