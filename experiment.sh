#!/bin/bash -e

python train.py --data 0.01 alex --transforms rotate10to15 --weights 0.001 --interpolate hamming
