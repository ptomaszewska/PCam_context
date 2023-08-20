#!/bin/bash

for value in pcamswin pcammoco pcamsup pcammae; do
   sbatch run_single.sh $value
done
