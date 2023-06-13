#!/bin/bash


export CUDA_VISIBLE_DEVICES="$1"
export dn="$2"
export tt="$3"
export fm=0
export to=1000

#sh att

cd /workspace/June/ADS/Finale_code/baselines/tmd-main/TextDefender/
for v in  70 100
do
  python main.py\
    --mode=attack\
    --model_type=bert\
    --dataset_name=$dn --dataset_path=/workspace/June/ADS/Finale_code/baselines/tmd-main/$dn/processed/ \
    --training_type=$tt \
    --modify_ratio=0.3 \
    --neighbour_vocab_size=$v\
    --do_lower_case=True --attack_method=bae --method=3 --start_index=$fm --end_index=$to
done
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

