# list of the packages we need for this program to run
import argparse
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from torch import nn, optim
from model import DNNModelClassifier
import utilities
from utilities import data_transforms as data_transforms
from torchvision import datasets, models


# add a short discription to help
parser = argparse.ArgumentParser(description='This is a short program to predict the labels from a pretrained CNN')

# the list of arguments which can be passed to the program
# set the amout of leading probabilities which are shown by the program
parser.add_argument('--top_k', type=int, default=5, help='top top_k results are shown [default: 5]')
# use a mapping of categories to real names
parser.add_argument('--category_names', default=None, help='use the provided json file to map categories to real names [default: None]')
# use GPU for the inference
parser.add_argument('--gpu', action='store_true', help='use GPU for inference [default: False]')
# show a picture of the flower and a bar plot
parser.add_argument('--visual', action='store_true', help='show a picture of the flower and a bar plot [default: False]')
# input image (possibly including path)
parser.add_argument('image_file', help='image file (including path) for which the inference is done')
# checkpoint to use for inference
parser.add_argument('checkpoint_file', help='checkpoint file of the model with which inference is done')

# for testing reasons we print the arguments of the parser
#print(parser.parse_args())
# assign the parser arguments to variables FLAGS for later use
FLAGS = parser.parse_args()

    
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

# read in the dictionary enoding the real labels relating to the directories
if FLAGS.category_names != None:
    # Check whether the category names file really exists otherwise raise an error
    if os.path.isfile(FLAGS.category_names):
        category_names_file = os.path.normpath(FLAGS.category_names)
        with open(category_names_file, 'r') as f:
            cat_to_name = json.load(f)
    else:
        raise ValueError("Couldn't not identify the category names file.")
    


# load the CNN from the checkpoint file
if os.path.isfile(FLAGS.checkpoint_file):
    checkpoint_file = os.path.normpath(FLAGS.checkpoint_file)
    dnn_model, _ = utilities.load_checkpoint(checkpoint_file, device)
else:
    raise ValueError("Couldn't not identify the checkpoint file.")

# load image and do the inference
if os.path.isfile(FLAGS.image_file):
    image_file = os.path.normpath(FLAGS.image_file)
    inference = utilities.predict(image_file, dnn_model, device, topk=FLAGS.top_k)
else:
    raise ValueError("Couldn't not identify the image file.")

# get real names of the infered classes
if FLAGS.category_names != None:
    inference[1] = [cat_to_name[x] for x in  inference[1]]
    
print('Inference of the model loaded from checkpoint: ', checkpoint_file)
for class_name, prob in zip(inference[1], inference[0]):
    print('The model predicts that the flower falls with {:.1f} percent certainty into class: {}'.format(prob*100, class_name))

# print imput image along with the top top_k most probable classes
if FLAGS.visual:
    utilities.image_predict(image_file, dnn_model, device, cat_to_name=cat_to_name, topk=FLAGS.top_k)

