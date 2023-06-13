#!/bin/bash



export CUDA_VISIBLE_DEVICES="$1"
export dn="$2"
export tt="$3"

cd /workspace/IGD/submission_to_github/TextDefender/
python main.py\
  --mode=train \
  --model_type=roberta \
  --ddp-backend=no_c10d\
  --model_name_or_path=/workspace/plm/roberta/\
  --dataset_name=$dn \
  --training_type=$tt \
  --do_lower_case=True\
