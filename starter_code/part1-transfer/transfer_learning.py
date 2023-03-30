# Starter code for Part 1 of the Small Data Solutions Project
# 

#Set up image data for train and test

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms 
from TrainModel import train_model
from TestModel import test_model
from torchvision import models
import os 

# use this mean and sd from torchvision transform documentation
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

#Set up Transforms (train, val, and test)

normalize = transforms.Normalize(
    mean=mean,
    std=std,
)

# define transforms
transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
])

input_size = 224
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(normalize)
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(normalize)
    ]),
    'test': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(normalize)
    ]),
}

#Set up DataLoaders (train, val, and test)
batch_size = 10
num_workers = 4

# load the dataset
data_dir = "./imagedata-50"

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val', 'test']}
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers) for x in ['train', 'val', 'test']}

#hint, create a variable that contains the class_names. You can get them from the ImageFolder
class_names = ['beach', 'desert', 'forest']


# Using the VGG16 model for transfer learning 
# 1. Get trained model weights
# 2. Freeze layers so they won't all be trained again with our data
# 3. Replace top layer classifier with a classifer for our 3 categories
num_classes = 3 
model = models.vgg16(pretrained=True)
for param in model.parameters():
        param.requires_grad = False
num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_ftrs,num_classes)

# Train model with these hyperparameters
# 1. num_epochs 
# 2. criterion 
# 3. optimizer 
# 4. train_lr_scheduler 

num_epochs = 15
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=100)
train_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

# When you have all the parameters in place, uncomment these to use the functions imported above
def main():
  trained_model = train_model(model, criterion, optimizer, train_lr_scheduler, dataloaders_dict['train'], dataloaders_dict['val'], num_epochs=num_epochs)
  test_model(dataloaders_dict['test'], trained_model, class_names)

if __name__ == '__main__':
   main()
   print("done")