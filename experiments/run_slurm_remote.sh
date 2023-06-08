#!/bin/bash

NUM_F=5
EXPERIMENT='experiments/exp_BRAIN_debug.py'

ssh fleet "ssh vm-biomedia-slurm 'bash /vol/medic01/users/mb4617/deployed/multitask-method/experiments/train_slurm_mm_all.sh ${EXPERIMENT} ${NUM_F}'"
