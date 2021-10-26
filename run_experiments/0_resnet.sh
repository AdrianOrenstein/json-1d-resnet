#!/bin/bash

rc=0;
counter=0;

testing_kwargs=(
    --limit_train_batches 51 
    --limit_val_batches 51 
    --limit_test_batches 51 
    --max_epochs 1 
)

resnet_experiment=(
    --experiment_name 'resnet'
    --experiment_description '' 
    --max_epochs 12
    --learning_rate 0.003
    --batch_size 64 
    --sequence_length 16384
    --balanced_dataset 'True'
)

python src/train.py --SEED 1 "${project_spec[@]}" "${resnet_experiment[@]}" #> ./mlruns/resnet_experiment-${counter}.txt 2>&1 || let 'rc += 1 << $counter'; let counter+=1; clear; make stop;
