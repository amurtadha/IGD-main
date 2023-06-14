 
 
 # IGD 
 An NLP advserarial defense
 
 This is the source code for the paper: "Integrated Gradients-based Defense against Adversarial Word Substitution Attacks".: 
  ```
  ```
 


# Requirements:
Required packages are listed in the requirements.txt file:

```
pip install -r requirements.txt
```
# Training

*  Go to scripts/         
*  Run the following code to train RNT:
```
sh training.sh 0 sst-2
```
* or directly use 
```
python main.py --dataset sst-2  --method train  --batch-size-generate 4 --n_steps_interpret 40 --num_epoch 10
```


# Evaluation the robutness 

*  Go to TextDefender\         
*  Run the following code:
```
  python main.py\
      --mode=attack\
      --model_type=bert\
      --dataset_name=sst-2 
      --dataset_path=datasets/sst-2/processed/ \
      --training_type=ours \
      --modify_ratio=0.3 \
      --do_lower_case=True \
      --attack_method=textfooler \
      --method=3 \
      --start_index=0 \
      --end_index=1000
```
- The params could be :
    - --dataset =\{sst-2, imdb\}
    - --attack_method ={textfooler, textbugger, bae}
