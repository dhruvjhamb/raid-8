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
    im_height, im_width = 64, 64

    ckpt = torch.load('./checkpoints/resnet/1586833409.8824692-1.pt', map_location=device)
    # model = Net(len(CLASSES), im_height, im_width)
    model = str_to_net[ckpt['model']](len(CLASSES), im_height, im_width)
    model.load_state_dict(ckpt['net'])
    model.eval()

    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0, 0, 0), tuple(np.sqrt((255, 255, 255)))),
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
            outputs = model(img)
            _, predicted = outputs.max(1)

            # Write the prediction to the output file
            eval_output_file.write('{},{}\n'.format(image_id, CLASSES[predicted]))


if __name__ == '__main__':
    main()
