# list of the packages we need for this program to run
import argparse
import os
import json
import numpy as np
import time
import torch
from torch import nn, optim
from model import DNNModelClassifier
import utilities
from utilities import data_transforms as data_transforms
from torchvision import datasets, models


# add a short discription to help
parser = argparse.ArgumentParser(description='This is a short program to adapt a pretrained CNN to new labels and train it')

# the list of arguments which can be passed to the program
# the directory to which checkpoints are saved
parser.add_argument('--save_dir', default='./', help='To set directory to save checkpoints')
# the architecture of the pretrained CNN
parser.add_argument('--arch', default='vgg13', help='To choose architecture from pretrained pytorch models')
# the learning rate which is used for training
parser.add_argument('--learning_rate', type=float, default=0.001, help='To set the learning rate [default: 0.001]')
# the hidden units in the classifier unit
parser.add_argument('--hidden_units', type=int, default=512, help='To sets the hidden units of the classifier [default: 512]')
# epochs to train the classifier
parser.add_argument('--epochs', type=int, default=6, help='To sets the epochs to train [default: 6]')
# to set the device to GPU
parser.add_argument('--gpu', action='store_true', default=False, help='Runs computation in GPU if available [default: False]')
# set the data/train directory
parser.add_argument('data_directory')

# for testing reasons we print the arguments of the parser
#print(parser.parse_args())
# assign the parser arguments to variables FLAGS for later use
FLAGS = parser.parse_args()



# Check whether the data_directory exists otherwise raise a error
if os.path.isdir(FLAGS.data_directory):
    data_dir = os.path.normpath(FLAGS.data_directory)
else:
    raise ValueError("Couldn't not identify the data_directory.")

    
# specify the directories of the training, testing, and validation set    
train_dir = data_dir + '/train'
test_dir = data_dir + '/test'
valid_dir = data_dir + '/valid'
data_dirs = {'train': train_dir, 'test': test_dir, 'valid': valid_dir}
                           

    
# Load the datasets with ImageFolder
image_datasets = {}
for set_i in ['train', 'test', 'valid']:
    image_datasets[set_i] = datasets.ImageFolder(data_dirs[set_i], transform=data_transforms[set_i])

# Using the image datasets and the transforms, define the dataloaders
dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True)}
for set_i in ['test', 'valid']:
    dataloaders[set_i] =  torch.utils.data.DataLoader(image_datasets[set_i], batch_size=32)
                   
# set device on which computations are done
if FLAGS.gpu:
    if torch.cuda.is_available():
        device = torch.device("cuda:0" )
    else:
        device = torch.device("cpu")
        print('GPU is not available')
else:
    device = torch.device("cpu")
print('Calculations are done on: ', device)




# the architecture used for the feature detection of the features
model_arch = FLAGS.arch
# initializing the CNN class
dnn_model = eval('models.'+ model_arch + '(pretrained=True)')
# Freeze parameters so we don't backprop through them
for param in dnn_model.parameters():
    param.requires_grad = False


# some parameters for the new classifier
try:
    iter(dnn_model.classifier)
    num_input = dnn_model.classifier[0].in_features
except TypeError:
    num_input = dnn_model.classifier.in_features
    
num_hid = FLAGS.hidden_units
num_output = len(image_datasets['train'].classes)
# replace the classifier of the chosen CNN model with the new classifier
dnn_model.classifier = DNNModelClassifier(num_input, num_output, num_hid)

# send the model to the device 
dnn_model.to(device)

# use the negative log likelihood loss because the output of classifier is log softmax
criterion = nn.NLLLoss()
# only train model on classifier parameters, feature parameters are frozen
optimizer = optim.Adam(dnn_model.classifier.parameters(), lr=FLAGS.learning_rate)

# train the classifier of the model
utilities.train_model(dnn_model, optimizer, criterion, dataloaders, device, num_epochs=FLAGS.epochs, print_every=2)

# save the class to index dictionary to the model
dnn_model.class_to_idx = image_datasets['train'].class_to_idx

# save a checkpoint of the model
utilities.save_checkpoint(dnn_model, model_arch, optimizer, num_input, num_hid, num_output, save_dir=FLAGS.save_dir)



