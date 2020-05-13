import sys
import pathlib
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from model import *

# current todos:
# not sure if allowed to import models from different file?
# ensemble will need to be implemented here

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load the classes
    data_dir = pathlib.Path('./data/tiny-imagenet-200/train/')
    CLASSES = sorted([item.name for item in data_dir.glob('*')])
    im_height, im_width = 224, 224

    checkpoints = ['./ensemble/topaz-base.pt',
            './ensemble/topaz-flip.pt',
            './ensemble/topaz-jitter.pt',
            './ensemble/topaz-rotate.pt',
            './ensemble/topaz-shear.pt'
            ]
    weights = [0.2,0.2,0.2,0.2,0.2]
    models = []
    for cpt in checkpoints:
            ckpt = torch.load(cpt, map_location=device)
            model = str_to_net[ckpt['model']](len(CLASSES), im_height, im_width, None)

            model.load_state_dict(ckpt['net'])
            if device.type == 'cuda':
                model.to(device)
            model.eval()
            # print ("Number of parameters: {}, Time: {}, User: {}"
            #                 .format(ckpt['num_params'], ckpt['runtime'], ckpt['machine'])) 
            models.append(model)
            del ckpt


    data_transforms = transforms.Compose([
        transforms.Resize((im_height, im_width)),
        transforms.ToTensor(),
    ])

    # Loop through the CSV file and make a prediction for each line
    with open('eval_classified.csv', 'w') as eval_output_file:  # Open the evaluation CSV file for writing
        for line in pathlib.Path(sys.argv[1]).open():  # Open the input CSV file for reading
            image_id, image_path, image_height, image_width, image_channels = line.strip().split(
                ',')  # Extract CSV info

            print(image_id, image_path, image_height, image_width, image_channels)
            with open(image_path, 'rb') as f:
                img = Image.open(f).convert('RGB')
            img = data_transforms(img)[None, :]
            sum_probabilities = None
            for i in range(len(models)):
                outputs = models[i](img.to(device))
                probabilites = (nn.Softmax(dim=-1)(outputs)).to(device)
                weighted_prob = (probabilites*weights[i]).to(device)
                if sum_probabilities is None:
                    sum_probabilities = weighted_prob
                else:
                    sum_probabilities = sum_probabilities + weighted_prob
                    sum_probabilities = sum_probabilities.to(device)
           
            _, predicted = sum_probabilities.max(1)

            # Write the prediction to the output file
            eval_output_file.write('{},{}\n'.format(image_id, CLASSES[predicted]))


if __name__ == '__main__':
    main()
