# World Models

![](assets/f0.gif)

A Tensorflow 2.0 reimplementation of *World Models* - David Ha, Jürgen Schmidhuber (2018).

You can find more detail about the project in this blog post](https://adgefficiency.com/world-models).  Resources and references are in [rl-resources/world-models](https://github.com/ADGEfficiency/rl-resources/tree/master/world-models).

##  Setup

```bash
git clone https://github.com/ADGEfficiency/world-models
```

## Using a pretrained agent

Download the pretrained vision, memory and controller (generation 229):

```bash
bash tf-cpu-setup.sh
bash pretrained.sh
```

To render fully, run the notebook `worldmodels/notebooks/render.ipynb`.

To sample data from the controller into `.npy` files in `~/world-models-experiments/controller-samples` (used for checking agent performance across random seeds):

```bash
python worldmodels/data/sample_policy.py --policy controller --dtype numpy --episode_length 1000 --num_process 4 --episodes 200 --generation 229
```

To sample from the controller into `.tfrecord` files in `~/world-models-experiments/controller-samples` (used to generate training data for the next iteration of agent):

```bash
python worldmodels/data/sample_policy.py --policy controller --dtype tfrecord --episode_length 1000 --num_process 4 --episodes 200 --generation 229
```

## Training from scratch

### Sample data using a random policy

A dataset is generated by from the environment using a random policy - data is placed into `$HOME/world-models-experiments/random-rollouts`.  The original paper uses 10,000 total episodes, with a max episode length of 1,000.  The dataset generation is parallelized using Python's `multiprocessing`.

To run the dataset generation (tested on Ubuntu 18.04.2 -  c5.4xlarge 512 GB storage):

```bash
bash gym-setup.sh

xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python3 worldmodels/data/sample_policy.py --num_process 8 --total_episodes 10000 --policy random

aws s3 sync ~/world-models-experiments/random-rollouts/ s3://world-models/random-rollouts
```

### Training the Variational Auto-Encoder (VAE)

Original paper uses 1 epoch, the code based supplied uses 10.

The autoencoder saves a copy of the model into `~/world-models-experiments/vae-training/models`.  Run on GPU:

To run the VAE training (tested on Ubuntu 18.04.02 - p3.2xlarge 512)

```bash
source tf-setup.sh
before_reboot
source tf-setup.sh
after_reboot

aws s3 sync s3://world-models/random-rollouts ~/world-models-experiments/random-rollouts

python3 worldmodels/vision/train_vae.py --load_model 0 --data local

nvidia-smi -l 1

tail -f ~/world-models-experiments/vae-training/training.csv

aws s3 sync ~/world-models-experiments/vae-training s3://world-models/vae-training
```

### Sampling latent statistics

Sample the statistics (mean & variance) of the VAE so we can generate more samples of the latent variables.  Run on CPU:

```bash
bash tf-cpu-setup.sh

aws s3 sync s3://world-models/vae-training/models ~/world-models-experiments/vae-training/models

python3 worldmodels/data/sample_latent_stats.py --episode_start 0 --episodes 10000 --data local --dataset random

aws s3 sync ~/world-models-experiments/latent-stats  s3://world-models/latent-stats
```

### Training LSTM Gaussian mixture

Done on GPU - p3.2xlarge

```bash
#  load before & after reboot from tf-setup
source tf-setup.sh

before_reboot

after_reboot

python3 worldmodels/memory/train_memory.py

aws s3 sync ~/world-models-experiments/memory-training  s3://world-models/memory-training
```

### Training the CMA-ES linear controller

```bash
aws s3 sync s3://world-models/vae-training/models/ ~/world-models-experiments/vae-training/models

aws s3 sync s3://world-models/memory-training/models/ ~/world-models-experiments/memory-training/models

xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python3 worldmodels/control/train_controller.py

tail -f ~/world-models-experiments/control/rewards.log

aws s3 sync ~/world-models-experiments/control/ s3://world-models/control
```

### Training the second generation

The process for training the second iteration of the agent is given below.  The main difference is that data is samples from a controller, not a random policy.

```bash

xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python3 worldmodels/data/sample_policy.py --num_process 8 --total_episodes 10000 --policy controller --dtype tfrecord

python3 worldmodels/vision/train_vae.py --load_model 0 --data local --epochs 15 --dataset controller

python3 worldmodels/data/sample_latent_stats.py --episode_start 0 --episodes 10000 --data local --dataset controller

python3 worldmodels/memory/train_memory.py --load_model 0 --epochs 80

xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python3 worldmodels/control/train_controller.py

aws s3 sync ~/world-models-experiments/control/ s3://world-models/control
```
