# this file includes all the program parts which are related to the CNN
# list of packages need
import torch
from torch import nn
from torchvision import models
from collections import OrderedDict


class DNNModelClassifier(nn.Sequential):
    """ a classifier class """
    def __init__(self, num_input, num_output, num_hid, drop_rate=0.3):
        super(DNNModelClassifier, self).__init__(OrderedDict([
                          # architecture of the classifier
                          ('drop1', nn.Dropout(p=drop_rate)),
                          ('fc1', nn.Linear(num_input, num_hid)),
                          ('relu', nn.ReLU()),
                          ('drop2', nn.Dropout(p=drop_rate)),
                          ('fc2', nn.Linear(num_hid, num_output)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
