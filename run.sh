#!/bin/bash
module load anaconda/2022.10
# conda create --name pymarl python=3.7
source activate pymarl
pip install -r requirements.txt
pip install -e src/envs/lb-foraging
pip install -e src/envs/mpe/multi_agent_particle
pip install -e src/envs/smac
python src/main.py --config=qmix_macpro_rnn --task-config=pp
