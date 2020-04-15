# Transforms file
# Key:      name of transformation
# Value:    [tuple] (transform, frac. images to transform)
#           e.g.    (transforms.Compose(...), 0.2)
data_transforms = dict() 

data_transforms['flip'] = (transforms.Compose([
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.ToTensor(),
    ]),
    0.01
)
