#!/bin/bash

export python_version=3.11

if [ "$1" != "train" ] && [ "$1" != "inference" ]; then
    echo "arg must be one of ['train', 'inference']"
    exit 1
elif [ "$1" == "train" ]; then
    python_file=train
    yaml_file=train
else
    python_file=eval
    yaml_file=inference
fi

for name in best cos-dec cos-default cos-restart ft-linear
do
    python$python_version ./src/$python_file.py -c model/$name/$yaml_file.yaml -n $name
done

if [ "$python_file" == "inference" ]; then
do
    python$python_version ./src/ensemble.py -c ensemble/ensemble.yaml
done