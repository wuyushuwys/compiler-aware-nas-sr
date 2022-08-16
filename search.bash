#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Experiments

model_type='NAS_MODEL'  # NAS_MODEL / BASIC_MODEL

speed_target=100    # target latency in ms

width_epochs=10     # width only search epoch
epochs=20           # width+depth search epoch
finetune_epochs=30  # fine-tune epoch
num_patches=1000    # default 1000
train_batch_size=16 # default 16
lr_patch_size=48    # default 48

# arch
scale=2
num_blocks=16
num_residual_units=24

num_gpus=$(awk -F '[0-9]' '{print NF-1}' <<<"$CUDA_VISIBLE_DEVICES")
echo Using $num_gpus GPUs

now=$(date +'%b%d_%H_%M_%S')

experiment_name=$1

if [ -z $experiment_name ]; then
  job_dir=wdsr_b_x${scale}_${num_blocks}_${num_residual_units}_${now}
else
  job_dir=${experiment_name}_${now}
fi

printf '%s\n' "Job save in runs/$job_dir"

if [ -d "runs/$job_dir" ]; then
  printf '%s\n' "Removing runs/$job_dir"
  rm -rf "runs/$job_dir"
fi

############ ACTIVATE environment here  ##############
#source /home/$USER/miniconda3/etc/profile.d/conda.sh
#conda activate SR
######################################################

printf '%s\n' "Training Model on GPU ${CUDA_VISIBLE_DEVICES}"

python -m torch.distributed.run --nproc_per_node $num_gpus --master_port $(((RANDOM % 1000 + 5000))) search.py \
  --model_type $model_type \
  --speed_target $speed_target \
  --dataset div2k \
  --eval_datasets set5 set14 urban100 bsds100 \
  --num_blocks $num_blocks \
  --num_residual_units $num_residual_units \
  --scale $scale \
  --train_batch_size $train_batch_size \
  --num_patches $num_patches \
  --lr_patch_size $lr_patch_size \
  --width_epochs $width_epochs \
  --epochs $epochs \
  --finetune_epochs $finetune_epochs \
  --width_search \
  --pretrained \
  --distributed \
  --job_dir runs/$job_dir


