# Pytorch base module written in C language
这是中文ReadMe

English Version 正在锐意制作中
## 简介
项目包含用C语言实现的Pytorch中的各个基本模块，和Python实现的测试框架，可以通过配置config文件，添加修改测试用例。


## 项目结构
```angular2html
.
├── README.md
├── __init__.py
├── bin/
├── config
│   ├── __init__.py
│   ├── __pycache__
│   │   └── __init__.cpython-38.pyc
│   ├── function/
│   └── module/
├── export.py
├── img_generator.py
├── import_config.py
├── model
│   ├── function/
│   ├── module/
│   └── tensor.py
├── pth/
├── run.c
├── src/
├── test/
├── test.py
├── test.sh
├── train.py
└── xmake.lua
```

项目基于xmake编写的，CMake支持会在后续开发中考虑。

## 用法
在src中添加C语言相关模块实现，并在model添加该模块对应的Pytorch实现,向Config文件夹配置测试用例，并修改import_config.py为对应config文件路径，执行`bash test.sh`即可自动输出测试结果。

## TODO
- 使用当前框架完成CLIP的C语言实现