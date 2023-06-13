#!/bin/bash


export CUDA_VISIBLE_DEVICES="$1"
export dn="$2"
export tt="ours"
export fm=0
export to=1000
export mr=0.1
#sh att

cd /workspace/June/ADS/Finale_code/baselines/tmd-main/TextDefender/
for v in  2 3 5 6 7 8
do
  for t in textfooler pwws bae
  do
    python main.py\
      --mode=attack\
      --model_type=bert\
      --dataset_name=$dn --dataset_path=/workspace/June/ADS/Finale_code/baselines/tmd-main/$dn/processed/ \
      --training_type=$tt \
      --modify_ratio=$mr \
      --n_candidates=$v \
      --do_lower_case=True --attack_method=$t --method=3 --start_index=$fm --end_index=$to
  done
done
