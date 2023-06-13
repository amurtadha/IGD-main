

export CUDA_VISIBLE_DEVICES="$1"
cd ../
python main.py --dataset $2  --method train  --batch-size-generate 4 --n_steps_interpret 40 --num_epoch 5
