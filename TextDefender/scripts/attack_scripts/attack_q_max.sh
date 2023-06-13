#!/bin/bash


export CUDA_VISIBLE_DEVICES="$1"
export dn="$2"
export tt="$3"
export lm="$4"
export v="$5"
export fm=0
export to=1000


cd /workspace/IGD/baselines/tmd-main/TextDefender/

for q in 0.4 0.5
do
  for am in textfooler pwws bae
  do
    python main.py\
          --mode=attack\
          --model_type=bert\
          --dataset_name=$dn --dataset_path=/workspace/June/ADS/Finale_code/baselines/tmd-main/$dn/processed/ \
          --training_type=$tt \
          --modify_ratio=$q \
          --neighbour_vocab_size=50\
          --lamnd=$lm\
          --n_candidates=$v \
          --do_lower_case=True --attack_method=$am --method=3 --start_index=$fm --end_index=$to
  done
done

# c 0.3