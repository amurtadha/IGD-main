 
 
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

*  Go to scripts/  and run the following code to train IGD:
```
sh training.sh 0 sst-2
```
* Or directly use 
```
python main.py --dataset sst-2  --method train --plm roberta --batch-size-generate 4 --n_steps_interpret 40 --num_epoch 10 
```


# Evaluation the robutness 
* Go to scripts/ and run the following code:
```
sh attack.sh 0 sst-2
```
* Or directly go to TextDefender\  and run the following code:
```
  python main.py\
      --mode=attack\
      --model_type=roberta\
      --dataset_name=sst-2 
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
* The output will be saved in  TextDefender\log\
