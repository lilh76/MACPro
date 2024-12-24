# Multi-agent Continual Coordination via Progressive Task Contextualization

This repository contains implementation for Multi-agent Continual Coordination via Progressive Task Contextualization (MACPro).

## Environment Installation

Build the environment by running:

```
pip install -r requirements.txt
```

or

```
conda env create -f environment.yaml
```

Install the Level Based Foraging (LBF) environment by running:

```
pip install -e src/envs/lb-foraging
```

Install the Predator-Prey (PP) environment by running:

```
pip install -e src/envs/mpe/multi_agent_particle
```

Install the StarCraft Multi-Agent Challenge (SMAC) environment by running:

```
pip install -e src/envs/smac
```

## Run an experiment

```
python src/main.py --config=[Algorithm name] --task-config=[Benchmark name]
```

The config files act as defaults for an algorithm or benchmark. They are all located in `src/config`. `--config` refers to the config files in `src/config/algs` including MACPro and Finetuning. `--task-config` refers to the config files in `src/config/tasks`, including `lbf` as the LB-Foraging benchmark (https://github.com/semitable/lb-foraging), `pp` as the Predator-Prey benchmark (https://github.com/openai/multiagent-particle-envs),  and `marines, sz` as the StarCraft Multi-Agent Challenge benchmark (https://github.com/oxwhirl/smac).

All results will be stored in the `results` folder.

For example, run MACPro on PP benchmark:

```
python src/main.py --config=qmix_macpro_rnn --task-config=pp
```

Run Finetuning on Marines benchmark:

```
python src/main.py --config=qmix_attn --task-config=marines
```

## Publication

If you find this repository useful, please [cite our paper](https://ieeexplore.ieee.org/document/10562331):

```
@article{macpro,
  title   = {Multi-agent Continual Coordination via Progressive Task Contextualization},
  author  = {Lei Yuan and Lihe Li and Ziqian Zhang and Fuxiang Zhang and Cong Guan and Yang Yu},
  journal = {IEEE Transactions on Neural Networks and Learning Systems},
  year    = {2024}
}
```