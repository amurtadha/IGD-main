#!/bin/bash

sbatch train_yelppolarity_ascc_bert.sh
sbatch train_yelppolarity_dne_bert.sh
sbatch train_yelppolarity_safer_bert.sh

#sbatch train_yelppolarity_ascc_roberta.sh
#sbatch train_yelppolarity_dne_roberta.sh
#sbatch train_yelppolarity_safer_roberta.sh

#sbatch train_yelppolarity_ascc_xlnet.sh
#sbatch train_yelppolarity_dne_xlnet.sh
#sbatch train_yelppolarity_safer_xlnet.sh

