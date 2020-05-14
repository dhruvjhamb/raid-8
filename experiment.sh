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

# Unfreezing batchnorm layers 
# CUDA_VISIBLE_DEVICES=0 python train.py --data 15 resnext --batchnorm_lr 1e-5 --learningrates 0 1e-4 1e-3 --partitions 6 3 1 --decayrate 1 --decaycoeff 0.75 --logfile bn-6_3_1-1e5
# CUDA_VISIBLE_DEVICES=0 python train.py --data 15 resnext --batchnorm_lr 1e-4 --learningrates 0 1e-4 1e-3 --partitions 6 3 1 --decayrate 1 --decaycoeff 0.75 --logfile bn-6_3_1-1e4
# CUDA_VISIBLE_DEVICES=0 python train.py --data 15 resnext --batchnorm_lr 1e-3 --learningrates 0 1e-4 1e-3 --partitions 6 3 1 --decayrate 1 --decaycoeff 0.75 --logfile bn-6_3_1-1e3

# Train all layers 
#CUDA_VISIBLE_DEVICES=0 python train.py --data 20 resnext --learningrates 1e-5 1e-4 1e-3 --partitions 6 3 1 --decayrate 1 --decaycoeff 0.75 --logfile all-layers-1e5
#CUDA_VISIBLE_DEVICES=0 python train.py --data 20 resnext --learningrates 1e-6 1e-4 1e-3 --partitions 6 3 1 --decayrate 1 --decaycoeff 0.75 --logfile all-layers-1e6

# Circular learning rates
#CUDA_VISIBLE_DEVICES=1 python train.py --data 20 resnext --learningrates 1e-5 1e-4 1e-3 --partitions 6 3 1 --decayrate 1 --decaycoeff 0.75 --circular_lr 0.5 --logfile circular-50
#CUDA_VISIBLE_DEVICES=1 python train.py --data 20 resnext --learningrates 1e-5 1e-4 1e-3 --partitions 6 3 1 --decayrate 1 --decaycoeff 0.75 --circular_lr 0.1 --logfile circular-10

# Train with transforms
#CUDA_VISIBLE_DEVICES=1 python train.py --data 20 resnext --learningrates 1e-5 1e-4 1e-3 --partitions 6 3 1 --decayrate 1 --decaycoeff 0.75 --circular_lr 0.1 --transforms shear --logfile topaz-shear
#CUDA_VISIBLE_DEVICES=1 python train.py --data 20 resnext --learningrates 1e-5 1e-4 1e-3 --partitions 6 3 1 --decayrate 1 --decaycoeff 0.75 --circular_lr 0.1 --transforms flip --logfile topaz-flip
# CUDA_VISIBLE_DEVICES=0 python train.py --data 20 resnext --learningrates 1e-5 1e-4 1e-3 --partitions 6 3 1 --decayrate 1 --decaycoeff 0.75 --circular_lr 0.1 --transforms rotateWithin45 --logfile topaz-rotate
# CUDA_VISIBLE_DEVICES=0 python train.py --data 20 resnext --learningrates 1e-5 1e-4 1e-3 --partitions 6 3 1 --decayrate 1 --decaycoeff 0.75 --circular_lr 0.1 --transforms jitter --logfile topaz-jitter
# CUDA_VISIBLE_DEVICES=1 python train.py --data 20 resnext --learningrates 1e-5 1e-4 1e-3 --partitions 6 3 1 --decayrate 1 --decaycoeff 0.75 --circular_lr 0.1 --transforms blur --logfile topaz-blur


# CUDA_VISIBLE_DEVICES=1 python train.py --data 10 resnext --logfile tin-base
# CUDA_VISIBLE_DEVICES=1 python train.py --data 10 resnext --logfile tin-jitter --transforms jitter
# CUDA_VISIBLE_DEVICES=1 python train.py --data 10 resnext --logfile tin-rotate --transforms rotateWithin45
CUDA_VISIBLE_DEVICES=0 python train.py --data 10 resnext --logfile tin-shear --transforms shear
CUDA_VISIBLE_DEVICES=0 python train.py --data 10 resnext --logfile tin-flip --transforms flip

