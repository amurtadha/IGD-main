#!/bin/bash


export CUDA_VISIBLE_DEVICES="$1"
export dn="$2"
export tt="asscc"
export fm=0
export to=1000



cd /workspace/June/ADS/Finale_code/baselines/tmd-main/TextDefender/

for t in textfooler pwws bae
  do
    python main.py\
      --mode=attack\
      --model_type=bert\
      --dataset_name=$dn --dataset_path=/workspace/June/ADS/Finale_code/baselines/tmd-main/$dn/processed/ \
      --training_type=$tt \
      --modify_ratio=0.3 \
      --do_lower_case=True --attack_method=$t --method=3 --start_index=$fm --end_index=$to
  done

for t in textfooler pwws bae
  do
    python main.py\
      --mode=attack\
      --model_type=bert\
      --dataset_name=$dn --dataset_path=/workspace/June/ADS/Finale_code/baselines/tmd-main/$dn/processed/ \
      --training_type=$tt \
      --modify_ratio=0.1 \
      --do_lower_case=True --attack_method=$t --method=3 --start_index=$fm --end_index=$to
  done

#python main.py\
#  --mode=attack\
#  --model_type=bert\
#  --dataset_name=$dn --dataset_path=/workspace/June/ADS/Finale_code/baselines/scripts-main/$dn/processed/ \
#  --training_type=$tt \
#  --modify_ratio=0.3 \
#  --do_lower_case=True --attack_method=textfooler --method=3 --start_index=$fm --end_index=$to
#
#
#
#python main.py\
#  --mode=attack\
#  --model_type=bert\
#  --dataset_name=$dn --dataset_path=/workspace/June/ADS/Finale_code/baselines/scripts-main/$dn/processed/ \
#  --training_type=$tt \
#  --modify_ratio=0.3 \
#  --do_lower_case=True --attack_method=pwws --method=3 --start_index=$fm --end_index=$to
#
#
#python main.py\
#  --mode=attack\
#  --model_type=bert\
#  --dataset_name=$dn --dataset_path=/workspace/June/ADS/Finale_code/baselines/scripts-main/$dn/processed/ \
#  --training_type=$tt \
#  --modify_ratio=0.4 \
#  --do_lower_case=True --attack_method=textfooler --method=3 --start_index=$fm --end_index=$to
#
#
#python main.py\
#  --mode=attack\
#  --model_type=bert\
#  --dataset_name=$dn --dataset_path=/workspace/June/ADS/Finale_code/baselines/scripts-main/$dn/processed/ \
#  --training_type=$tt \
#  --modify_ratio=0.4 \
#  --do_lower_case=True --attack_method=pwws --method=3 --start_index=$fm --end_index=$to
#
#
#
#python main.py\
#  --mode=attack\
#  --model_type=bert\
#  --dataset_name=$dn --dataset_path=/workspace/June/ADS/Finale_code/baselines/scripts-main/$dn/processed/ \
#  --training_type=$tt \
#  --modify_ratio=0.5 \
#  --do_lower_case=True --attack_method=textfooler --method=3 --start_index=$fm --end_index=$to
#
#
#python main.py\
#  --mode=attack\
#  --model_type=bert\
#  --dataset_name=$dn --dataset_path=/workspace/June/ADS/Finale_code/baselines/scripts-main/$dn/processed/ \
#  --training_type=$tt \
#  --modify_ratio=0.5\
#  --do_lower_case=True --attack_method=pwws --method=3 --start_index=$fm --end_index=$to
#
#


#for m in  70 100
#do
#  python main.py\
#    --mode=attack\
#    --model_type=bert\
#    --dataset_name=$dn --dataset_path=/workspace/June/ADS/Finale_code/baselines/scripts-main/$dn/processed/ \
#    --training_type=$tt \
#    --modify_ratio=0.3 \
#    --neighbour_vocab_size=$m\
#    --do_lower_case=True --attack_method=bae --method=3 --start_index=$fm --end_index=$to
#done