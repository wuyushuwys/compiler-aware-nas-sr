#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Experiments

epochs=30
num_patches=1000    # default 1000
train_batch_size=16 # default 16
lr_patch_size=48    # default 48

# AE
## Encoder Parameters

## Decoder Parameters
scale=2
num_blocks=16
num_residual_units=24

num_gpus=$(awk -F '[0-9]' '{print NF-1}' <<<"$CUDA_VISIBLE_DEVICES")
echo Using $num_gpus GPUs

resume_folder=$1

printf '%s\n' "Job resume $resume_folder"

printf '%s\n' "Training Model on GPU ${CUDA_VISIBLE_DEVICES}"

python -m torch.distributed.run --nproc_per_node $num_gpus --master_port $(((RANDOM % 1000 + 5000))) train.py \
  --dataset div2k \
  --eval_datasets set5 \
  --decoder_blocks $num_blocks \
  --$num_residual_units $num_residual_units \
  --scale $scale \
  --train_batch_size $train_batch_size \
  --num_patches $num_patches \
  --lr_patch_size $lr_patch_size \
  --epochs $epochs \
  --distributed \
  --resume \
  --job_dir $resume_folder