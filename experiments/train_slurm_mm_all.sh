#!/bin/bash

for ((f=0;f<$2;f++)); do
  sbatch --mem=24G --gres gpu:1 -p gpus /vol/medic01/users/mb4617/deployed/multitask-method/experiments/train_slurm_mm.sh $1 $f
done