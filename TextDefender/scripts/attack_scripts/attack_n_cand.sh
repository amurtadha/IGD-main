#!/bin/bash


export CUDA_VISIBLE_DEVICES="$1"
export dn="$2"
export tt="ours"
export fm=0
export to="$3"
export mr="$4"
export lm="$5"
#sh attack.sh 3 imdb 1000 0.3 0.3 3s
cd /workspace/IGD/baselines/tmd-main/TextDefender/
for v in  $6
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
      --lamnd=$lm\
      --do_lower_case=True --attack_method=$t --method=3 --start_index=$fm --end_index=$to
  done
done
