#!/bin/bash

cd /vol/medic01/users/mb4617/deployed/multitask-method

source multitask_method_env/bin/activate

python train.py $1 $2 --overwrite_folder