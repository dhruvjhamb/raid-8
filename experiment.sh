#!/bin/bash -e

# Train only the FC layer 
# CUDA_VISIBLE_DEVICES=0 python train.py --data 10 resnext --learningrates 0 1e-3 --partitions 9 1 --logfile fc-train-1e3
# CUDA_VISIBLE_DEVICES=0 python train.py --data 10 resnext --learningrates 0 1e-3 --partitions 9 1 --logfile fc-train-1e3
# CUDA_VISIBLE_DEVICES=0 python train.py --data 10 resnext --learningrates 0 1e-3 --partitions 9 1 --logfile fc-train-1e3
# CUDA_VISIBLE_DEVICES=0 python train.py --data 10 resnext --learningrates 0 1e-3 --partitions 9 1 --logfile fc-train-1e3
# CUDA_VISIBLE_DEVICES=0 python train.py --data 10 resnext --learningrates 0 1e-3 --partitions 9 1 --logfile fc-train-1e3

# Add LR decay 
# CUDA_VISIBLE_DEVICES=1 python train.py --data 15 resnext --learningrates 0 1e-4 1e-3 --partitions 6 3 1 --decayrate 1 --decaycoeff 0.75 --logfile decay-6_3_1-75
# CUDA_VISIBLE_DEVICES=1 python train.py --data 15 resnext --learningrates 0 1e-4 1e-3 --partitions 6 3 1 --decayrate 1 --decaycoeff 0.5 --logfile  decay-6_3_1-50 
# CUDA_VISIBLE_DEVICES=1 python train.py --data 15 resnext --learningrates 0 1e-4 1e-3 --partitions 6 3 1 --decayrate 1 --decaycoeff 0.9 --logfile  decay-6_3_1-90
# CUDA_VISIBLE_DEVICES=1 python train.py --data 15 resnext --learningrates 0 1e-4 1e-3 --partitions 6 2 2 --decayrate 1 --decaycoeff 0.75 --logfile decay-6_2_2-75
# CUDA_VISIBLE_DEVICES=1 python train.py --data 15 resnext --learningrates 0 1e-4 1e-3 --partitions 6 2 2 --decayrate 1 --decaycoeff 0.5 --logfile  decay-6_2_2-50 
# CUDA_VISIBLE_DEVICES=1 python train.py --data 15 resnext --learningrates 0 1e-4 1e-3 --partitions 6 2 2 --decayrate 1 --decaycoeff 0.9 --logfile  decay-6_2_2-90

# Unfreezing batchnorm layers 
# CUDA_VISIBLE_DEVICES=0 python train.py --data 15 resnext --batchnorm_lr 1e-5 --learningrates 0 1e-4 1e-3 --partitions 6 3 1 --decayrate 1 --decaycoeff 0.75 --logfile bn-6_3_1-1e5
# CUDA_VISIBLE_DEVICES=0 python train.py --data 15 resnext --batchnorm_lr 1e-4 --learningrates 0 1e-4 1e-3 --partitions 6 3 1 --decayrate 1 --decaycoeff 0.75 --logfile bn-6_3_1-1e4
# CUDA_VISIBLE_DEVICES=0 python train.py --data 15 resnext --batchnorm_lr 1e-3 --learningrates 0 1e-4 1e-3 --partitions 6 3 1 --decayrate 1 --decaycoeff 0.75 --logfile bn-6_3_1-1e3
# CUDA_VISIBLE_DEVICES=0 python train.py --data 15 resnext --batchnorm_lr 1e-5 --learningrates 0 1e-4 1e-3 --partitions 6 2 2 --decayrate 1 --decaycoeff 0.75 --logfile bn-6_2_2-1e5
# CUDA_VISIBLE_DEVICES=0 python train.py --data 15 resnext --batchnorm_lr 1e-4 --learningrates 0 1e-4 1e-3 --partitions 6 2 2 --decayrate 1 --decaycoeff 0.75 --logfile bn-6_2_2-1e4
# CUDA_VISIBLE_DEVICES=0 python train.py --data 15 resnext --batchnorm_lr 1e-3 --learningrates 0 1e-4 1e-3 --partitions 6 2 2 --decayrate 1 --decaycoeff 0.75 --logfile bn-6_2_2-1e3

# Train all layers 
CUDA_VISIBLE_DEVICES=0 python train.py --data 20 resnext --learningrates 1e-5 1e-4 1e-3 --partitions 6 2 2 --decayrate 1 --decaycoeff 0.75 --logfile all-layers-1e5
CUDA_VISIBLE_DEVICES=0 python train.py --data 20 resnext --learningrates 1e-6 1e-4 1e-3 --partitions 6 2 2 --decayrate 1 --decaycoeff 0.75 --logfile all-layers-1e6

# Circular learning rates
CUDA_VISIBLE_DEVICES=1 python train.py --data 20 resnext --learningrates 1e-5 1e-4 1e-3 --partitions 6 2 2 --decayrate 1 --decaycoeff 0.75 --circular_lr 0.5 --logfile circular-50
CUDA_VISIBLE_DEVICES=1 python train.py --data 20 resnext --learningrates 1e-5 1e-4 1e-3 --partitions 6 2 2 --decayrate 1 --decaycoeff 0.75 --circular_lr 0.1 --logfile circular-50
