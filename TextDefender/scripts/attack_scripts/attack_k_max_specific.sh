#!/bin/bash


export CUDA_VISIBLE_DEVICES="$1"
export dn="$2"
export tt="$3"
export fm=0
export to=1000


cd /workspace/IGD/baselines/tmd-main/TextDefender/
for k in 70 100
do
  for am in textfooler pwws bae
  do
    python main.py\
          --mode=attack\
          --model_type=bert\
          --dataset_name=$dn --dataset_path=/workspace/June/ADS/Finale_code/baselines/tmd-main/$dn/processed/ \
          --training_type=$tt \
          --modify_ratio=0.3 \
          --neighbour_vocab_size=$k\
          --do_lower_case=True --attack_method=$am --method=3 --start_index=$fm --end_index=$to
  done
done


#sh attack_k_max.sh 0 imdb ours 70
#sh attack_k_max.sh 0 imdb ours 100 pwws