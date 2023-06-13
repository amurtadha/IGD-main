#!/bin/bash


export CUDA_VISIBLE_DEVICES="$1"
export dn="$2"
export tt="safer"
export fm=0
export em=100
export to=1000



cd /workspace/June/ADS/Finale_code/baselines/tmd-main/TextDefender/
python main.py\
  --mode=attack\
  --model_type=bert\
  --dataset_name=$dn --dataset_path=/workspace/June/ADS/Finale_code/baselines/tmd-main/$dn/processed/ \
  --training_type=$tt \
  --modify_ratio=0.3 \
  --do_lower_case=True --attack_method=textfooler --method=3 --start_index=$fm --end_index=$to



python main.py\
  --mode=attack\
  --model_type=bert\
  --dataset_name=$dn --dataset_path=/workspace/June/ADS/Finale_code/baselines/tmd-main/$dn/processed/ \
  --training_type=$tt \
  --modify_ratio=0.3 \
  --do_lower_case=True --attack_method=pwws --method=3 --start_index=$fm --end_index=$to


python main.py\
  --mode=attack\
  --model_type=bert\
  --dataset_name=$dn --dataset_path=/workspace/June/ADS/Finale_code/baselines/tmd-main/$dn/processed/ \
  --training_type=$tt \
  --modify_ratio=0.4 \
  --do_lower_case=True --attack_method=textfooler --method=3 --start_index=$fm --end_index=$to


python main.py\
  --mode=attack\
  --model_type=bert\
  --dataset_name=$dn --dataset_path=/workspace/June/ADS/Finale_code/baselines/tmd-main/$dn/processed/ \
  --training_type=$tt \
  --modify_ratio=0.4 \
  --do_lower_case=True --attack_method=pwws --method=3 --start_index=$fm --end_index=$to



python main.py\
  --mode=attack\
  --model_type=bert\
  --dataset_name=$dn --dataset_path=/workspace/June/ADS/Finale_code/baselines/tmd-main/$dn/processed/ \
  --training_type=$tt \
  --modify_ratio=0.5 \
  --do_lower_case=True --attack_method=textfooler --method=3 --start_index=$fm --end_index=$to


python main.py\
  --mode=attack\
  --model_type=bert\
  --dataset_name=$dn --dataset_path=/workspace/June/ADS/Finale_code/baselines/tmd-main/$dn/processed/ \
  --training_type=$tt \
  --modify_ratio=0.5\
  --do_lower_case=True --attack_method=pwws --method=3 --start_index=$fm --end_index=$to


# k_max
#python main.py\
#  --mode=attack\
#  --model_type=bert\
#  --dataset_name=$dn --dataset_path=/workspace/June/ADS/Finale_code/baselines/scripts-main/$dn/processed/ \
#  --training_type=$tt \
#  --modify_ratio=0.3 \
#  --neighbour_vocab_size=70\
#  --do_lower_case=True --attack_method=textfooler --method=3 --start_index=$fm --end_index=$to
#
#
#python main.py\
#  --mode=attack\
#  --model_type=bert\
#  --dataset_name=$dn --dataset_path=/workspace/June/ADS/Finale_code/baselines/scripts-main/$dn/processed/ \
#  --training_type=$tt \
#  --modify_ratio=0.3 \
#  --neighbour_vocab_size=70\
#  --do_lower_case=True --attack_method=pwws --method=3 --start_index=$fm --end_index=$to
#
#python main.py\
#  --mode=attack\
#  --model_type=bert\
#  --dataset_name=$dn --dataset_path=/workspace/June/ADS/Finale_code/baselines/scripts-main/$dn/processed/ \
#  --training_type=$tt \
#  --modify_ratio=0.3 \
#  --neighbour_vocab_size=100\
#  --do_lower_case=True --attack_method=textfooler --method=3 --start_index=$fm --end_index=$to
#
#
#python main.py\
#  --mode=attack\
#  --model_type=bert\
#  --dataset_name=$dn --dataset_path=/workspace/June/ADS/Finale_code/baselines/scripts-main/$dn/processed/ \
#  --training_type=$tt \
#  --modify_ratio=0.3\
#  --neighbour_vocab_size=100\
#  --do_lower_case=True --attack_method=pwws --method=3 --start_index=$fm --end_index=$to
#
#
#
