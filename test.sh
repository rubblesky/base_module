#!/bin/bash
xmake f -m release
xmake 

python train.py
python export.py
python test.py

xmake f -m debug