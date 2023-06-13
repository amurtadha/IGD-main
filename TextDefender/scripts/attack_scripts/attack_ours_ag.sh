#!/bin/bash


export CUDA_VISIBLE_DEVICES="$1"
export dn="agnews"
export tt="ours"
export fm=0
export to=1000



cd /workspace/June/ADS/Finale_code/baselines/tmd-main/TextDefender/

#for i in 50 70 100
#do
#  for at in textfooler pwws bae
#  do
#    python main_k_max.py --dataset $2 --attack_method $at  --method attack  --n_candidates 4  --neighbour_vocab_size  $i
#  done
#done
for at in textfooler pwws bae
  do
  python main.py\
    --mode=attack\
    --model_type=bert\
    --dataset_name=$dn --dataset_path=/workspace/June/ADS/Finale_code/baselines/tmd-main/$dn/processed/ \
    --training_type=$tt \
    --modify_ratio=0.3 \
    --do_lower_case=True --attack_method=$at --method=3 --start_index=$fm --end_index=$to
done
