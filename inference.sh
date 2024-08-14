#!/bin/bash
# Script for inference

export py_ver=3.11

for name in best cos-dec cos-default cos-restart ft-linear
do
    python$py_ver ./src/eval.py -c model/$name/inference.yaml -n $name
done