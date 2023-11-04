typedef struct LinearModule
{
    float * weight;
    float * bias;
    int input_size;
    int output_size;
}LinearModule;
LinearModule * create_linear_module(int input_size, int output_size) ;
void load_linear_module(LinearModule * linear_module, char * path);
void free_linear_module(LinearModule * linear_module);
void forward_linear_module(LinearModule * linear_module, float * input,float * output);

void * build_linear_module(char * path,int input_size,int output_size);
float * test_linear_module(void* lm,float * input);

void test_input(float * input,int input_size);