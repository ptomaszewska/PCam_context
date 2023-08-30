#!/bin/bash
mkdir pretrained_experiments
for value in pcamswin pcammoco pcamsup pcammae resnet18-pcam densenet121-pcam; do
   sbatch run_single.sh $value
done
