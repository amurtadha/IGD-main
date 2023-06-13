#!/bin/bash


export CUDA_VISIBLE_DEVICES="$1"
export dn="$2"
export tt="ours"
export fm=0
export to="$3"
export mr="$4"
#sh attack_without_lm.sh 3 imdb 100 0.1

cd /workspace/IGD/baselines/tmd-main/TextDefender/

for t in textfooler pwws bae
  do
    python main.py\
      --mode=attack\
      --model_type=bert\
      --dataset_name=$dn --dataset_path=/workspace/June/ADS/Finale_code/baselines/tmd-main/$dn/processed/ \
      --training_type=$tt \
      --modify_ratio=$mr \
      --n_candidates=4 \
      --inactivate_lm_loss=1\
      --inactivate_data_loss=1\
      --do_lower_case=True --attack_method=$t --method=3 --start_index=$fm --end_index=$to

done

