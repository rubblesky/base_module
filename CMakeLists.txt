cmake_minimum_required(VERSION 3.26)
project(base_module C)

set(CMAKE_C_STANDARD 11)

include_directories(src/library)

add_executable(base_module
        src/library/batchnorm.c
        src/library/batchnorm.h
        src/library/conv.c
        src/library/conv.h
        src/library/data.c
        src/library/data.h
        src/library/layernorm.c
        src/library/layernorm.h
        src/library/linear.c
        src/library/linear.h
        src/library/op.c
        src/library/op.h
        src/library/tensor.c
        src/library/tensor.h
        run.c)
