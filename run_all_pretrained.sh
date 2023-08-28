#!/bin/bash
mkdir pretrained_experiments
for value in pcamswin pcammoco pcamsup pcammae; do
   sbatch run_single.sh $value
done
