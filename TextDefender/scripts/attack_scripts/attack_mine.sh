#!/bin/bash
#SBATCH --partition=applied
#SBATCH --output=/lustre/scratch/client/vinai/users/dangnm12/slurm/log/%x-%j.out
#SBATCH --error=/lustre/scratch/client/vinai/users/dangnm12/slurm/log/%x-%j.out
#SBATCH --job-name=yelppolarity_safer_bert_pwws_None_recmethod_3_from0_to334
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --mail-user=v.dangnm12@vinai.io
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

export CUDA_VISIBLE_DEVICES="$1"
export dn="$2"
export am="$3"
export tt="$4"
export mr="$5"
export fm=0
export to=1000


cd /workspace/June/ADS/Finale_code/baselines/tmd-main/TextDefender/
python main.py\
  --mode=attack\
  --model_type=bert\
  --dataset_name=$dn --dataset_path=/workspace/June/ADS/Finale_code/baselines/tmd-main/$dn/processed/ \
  --training_type=$tt \
  --modify_ratio=$mr\
  --do_lower_case=True --attack_method=$am --method=3 --start_index=$fm --end_index=$to

#sh attack_mine.sh 1 imdb textfooler ours  0.3
#srun --container-image=/lustre/scratch/client/vinai/users/dangnm12/setup/docker_images/dc-miniconda3-py:38-4.10.3-cuda11.4.2-cudnn8-ubuntu20.04.sqsh \
#     --container-mounts=/lustre/scratch/client/vinai/users/dangnm12/:/root/ \
#     --container-workdir=/root/ \
#     /bin/bash -c \
#     "
#     export HTTP_PROXY=http://proxytc.vingroup.net:9090/
#     export HTTPS_PROXY=http://proxytc.vingroup.net:9090/
#     export http_proxy=http://proxytc.vingroup.net:9090/
#     export https_proxy=http://proxytc.vingroup.net:9090/
#
#     export HF_DATASETS_CACHE=/root/cache/huggingface/datasets
#     export TRANSFORMERS_CACHE=/root/cache/huggingface/transformers
#     export TFHUB_CACHE_DIR=/root/cache/tfhub_modules
#
#     export TF_FORCE_GPU_ALLOW_GROWTH=true
#     export TRANSFORMERS_OFFLINE=0
#     export HF_DATASETS_OFFLINE=0
#     export TOKENIZERS_PARALLELISM=false
#
#     source /opt/conda/bin/activate
#
#     cd /root
#     conda activate /root/miniconda3/envs/textdef
#
#     cd /root/TextDefender
#     python main.py --mode=attack --model_type=bert --model_name_or_path=/root/manifold_defense/models/bert-base-uncased-yelppolarity --dataset_name=yelppolarity --dataset_path=/root/TextDefender/dataset/yelppolarity --training_type=safer --max_seq_len=256 --do_lower_case=True --attack_method=pwws --method=3 --start_index=0 --end_index=334
#     "