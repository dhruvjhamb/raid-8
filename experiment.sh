#!/bin/bash -e

python train.py --data 15 resnext --learningrates 0 1e-4 1e-3 --partitions 3 1 1 --decayrate 1 --decaycoeff 0.75 --logfile resnext_3fc_2relu_2drop
