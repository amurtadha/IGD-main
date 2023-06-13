
export CUDA_VISIBLE_DEVICES="$1"
export dn="$2"
export tt="$3"



cd /workspace/June/ADS/Finale_code/baselines/tmd-main/TextDefender/

python main.py \
    --mode=evaluate \
    --model_type=bert\
    --dataset_name=$dn \
    --training_type=$tt \
    --do_lower_case=True \
    --method=3

