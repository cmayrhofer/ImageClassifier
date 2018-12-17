# Image Classifier
--------
In this repository, we provide the code implementation for an image classifier which can be trained on any image data set but is pretrained on [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, you can see a few examples below. 

<img src='Flowers.png' width=500px>



<!-- Now that you've built and trained a deep neural network on the flower data set, it's time to convert it into an application that others can use. Your application should be a pair of Python scripts that run from the command line. For testing, you should use the checkpoint you saved in the first part. Specifications

The project submission must include at least two files train.py and predict.py. The first file, train.py, will train a new network on a dataset and save the model as a checkpoint. The second file, predict.py, uses a trained network to predict the class for an input image. Feel free to create as many other files as you need. Our suggestion is to create a file just for functions and classes relating to the model and another one for utility functions like loading data and preprocessing images. Make sure to include all files necessary to run train.py and predict.py in your submission.
-->

# How to run the Classifier
------

To train a new network on a data set run `train.py`:
* Basic usage: `python train.py data_directory`  
  Prints out training loss, validation loss, and validation accuracy as the network

* Options:  
  Set directory to save checkpoints: `python train.py data_dir --save_dir save_directory`  
  Choose architecture: `python train.py data_dir --arch "vgg13"`  
  Set hyperparameters: `python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20`  
  Use GPU for training: `python train.py data_dir --gpu`

To predict the flower name from an image run `predict.py`, it will print out the most probable names along with the probabilities of that names.  
* Basic usage: `python predict.py /path/to/image checkpoint`  
* Options:  
  Return top KKK most likely classes: `python predict.py input checkpoint --top_k KKK`  
  Use a mapping of categories to real names: `python predict.py input checkpoint --category_names cat_to_name.json`  
  Use GPU for inference: `python predict.py input checkpoint --gpu`

# Requirements 
------

To run the code provided in this repository, you must have python 3.6 or higher installed. In addition, you will need to have the packages: `numpy`, `matplotlib`, `torch`, `torchvision`, `argparse`, `json`, and the dependencies of the packages installed. 
