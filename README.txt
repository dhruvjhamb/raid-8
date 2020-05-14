This is code for our CS 182/282 Computer Vision project (PyTorch). It has the following files:

README.txt - This file
requirements.txt - The python requirments necessary to run this project
train.py - A training file which trains a specified model on the data, and saves the checkpoint to be loaded
test_submission.py - A file which will return an output for every input in the eval.csv
eval.csv - An example test file
data/get_data.sh - A script which will download the tiny-imagenet data into the data/tiny-imagenet-200 file
ensemble/ - Contains checkpoints for models in the final ensemble
model.py - Contains implemented model architectures
synthetic_data.py - Applies transforms found in transforms.py to datasets to generate perturbed images
transform.py - Contains transformations to be applied on the data

Note: You should be using Python 3.8 to run this code.

Instructions:
1. In data directory, run get_data.sh
2. Run train_formatter.py
3. Run val_formatter.py to organize the validation data in a similar format to the training data
4. Run train.py to train a model and save a checkpoint

For example, the following command trains a ResNeXt model with the specified parameters on dataset containing unperturbed, flipped, and sheared training images:
python train.py --data 20 resnext --learningrates 1e-5 1e-4 1e-3 --partitions 6 3 1 --decayrate 1 --decaycoeff 0.75 --circular_lr 0.1 --transforms flip shear
