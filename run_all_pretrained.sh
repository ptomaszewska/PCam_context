#!/bin/bash

#for value in alexnet-pcam densenet121-pcam googlenet-pcam resnet101-pcam resnet18-pcam; do
#for value in densenet121-pcam resnet18-pcam pcamswin pcammoco pcamsup pcammae; do
for value in pcamswin pcammoco pcamsup pcammae; do
   sbatch run_single.sh $value
done
