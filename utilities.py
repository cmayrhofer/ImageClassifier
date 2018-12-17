# import packages needed in the following
import torch
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from PIL import Image
from torch import nn, optim
from torchvision import datasets, transforms, models



# transforms for the training, validation, and testing sets
data_transforms = {'train': transforms.Compose(
                            [transforms.RandomResizedCrop(224, scale=(0.33, 1.0)),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomVerticalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                   'test': transforms.Compose([transforms.Resize(256), 
                           transforms.CenterCrop(224),
                           transforms.ToTensor(),
                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
# use the same transformations for validation and test
data_transforms['valid'] = data_transforms['test']

def validation(model, testloader, criterion, device):
    """ a validation function which returns the loss and accuaracy over the whole test dataset
    
    Input:
    ======
        model(nn.Module): pytorch neural network
        testloader:
        criterion: loss criterion to be used
        device: device on which the computations are done (cpu/gpu)
    """

    accuracy = 0
    test_loss = 0
    model.eval()
    with torch.no_grad():
        for images, labels in testloader:
        # Move input and label tensors to the GPU
            images, labels = images.to(device), labels.to(device)

            output = model.forward(images)
            test_loss += criterion(output, labels)

            ## Calculating the accuracy 
            # Model's output is log-softmax, take exponential to get the probabilities
            ps = torch.exp(output)
            # Class with highest probability is our predicted class, compare with true label
            equality = (labels.data == ps.max(1)[1])
            # Accuracy is number of correct predictions divided by all predictions, just take the mean
            accuracy += equality.type_as(torch.FloatTensor()).mean()
    
    model.train()
    return test_loss/len(testloader), accuracy/len(testloader)
    
    
    
def train_model(model, optimizer, criterion, dataloaders, device, num_epochs=6, print_every=40):
    """function to train the model nodel
    
    Input:
    ======
        optimizer
        criterion: pytorch loss function
        dataloaders(dict): dictionary of dataloaders with keys: 'train' and 'test'
        deveice: device on which calculations are done
    Params:
    =======
        num_epochs: number of epochs to train for
        print_every: amount of steps before info is printed out
    """
    for epoch in range(num_epochs):
        # initialize start time and running loss
        start_time = time.time()
        running_loss = 0
        for step, (inputs, labels) in enumerate(dataloaders['train']):

            # Move input and label tensors to device
            inputs, labels = inputs.to(device), labels.to(device)

            # set the optimizer to zero
            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss


            if (step + 1) % print_every == 0:
                test_loss, accuracy = validation(model, dataloaders['test'], criterion, device)
                print("Epoch: {}/{}, Steps: {}, Passed Time: {:.2f} mins, ".format(\
                       epoch + 1, num_epochs, step + 1, (time.time() - start_time)/60.),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Test Loss: {:.3f}.. ".format(test_loss),
                      "Test Accuracy: {:.3f}".format(accuracy))
                start_time = time.time()
                running_loss = 0

def save_checkpoint(model, model_arch, optimizer, num_input, num_hid, num_output, drop_rate=0.3, save_dir='./'):
    """saves a checkpoint of the given model to location save_dir
    
    Input:
    =====
        model
        model_arch
        optimizer
    """
    checkpoint = {'ModelArch': model_arch,
              'classifier': {'num_input': num_input,
                             'num_hid': num_hid,
                             'num_output': num_output,
                             'drop_rate': drop_rate},
              'state_dict': model.state_dict(),
              'state_dict_optimizer': optimizer.state_dict(),
              'class_to_idx': model.class_to_idx}
    path_to_checkpoint = os.path.normpath(save_dir) + '/checkpoint_of_' + model_arch + '_full_info.pth'
    torch.save(checkpoint, path_to_checkpoint)
    print('Model saved to: ', path_to_checkpoint)


def load_checkpoint(filepath, device):
    """Returns the model and the optimizer given a checkpoint 
    
    Params
    ======
        filepath (string): file including path where checkpoint is stored
        device (string): the devise to which the state_dict should be mapped
    """
    
    checkpoint = torch.load(filepath) #, map_location=device)

    model = eval('models.' + checkpoint['ModelArch'] + '(pretrained=True)')
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                          ('drop1', nn.Dropout(p=checkpoint['classifier']['drop_rate'])),
                          ('fc1', nn.Linear(checkpoint['classifier']['num_input'], checkpoint['classifier']['num_hid'])),
                          ('relu', nn.ReLU()),
                          ('drop2', nn.Dropout(p=checkpoint['classifier']['drop_rate'])),
                          ('fc2', nn.Linear(checkpoint['classifier']['num_hid'], checkpoint['classifier']['num_output'])),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = optim.Adam(model.classifier.parameters())
    optimizer.load_state_dict(checkpoint['state_dict_optimizer'])
    
    return (model, optimizer)


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a tensor
        
        Input:
        ======
            image: PIL image
            trafo: pytorch transformation which should be applied
    '''
    
    # Process a PIL image for use in a PyTorch model
    return data_transforms['valid'](image)
    
def predict(image_path, model, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # predict the class from an image file
    image = Image.open(image_path)
    image = process_image(image).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        # Move input and label tensors to the GPU
        image = image.to(device)

        output = model.forward(image)
        # Model's output is log-softmax, take exponential to get the probabilities
        ps = torch.exp(output)
    model.train()
    # get the invers map of class_to_idx
    idx_to_class = dict([(v, k) for (k, v) in model.class_to_idx.items()])
    inference = [i.numpy()[0] for i in ps.topk(topk)]
    # apply the idx_to_class dictionary to obtain the index labels of the data
    inference[1] =  np.vectorize(idx_to_class.get)(inference[1])
    return inference
        
        

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
        
    if title is not None:
        ax.set_title(title)
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.set_axis_off()
    ax.autoscale(enable=True, axis='both', tight=True)
    ax.imshow(image)
    
    return ax
    
# Display an image along with the top 5 classes
def image_predict(image_path, model, device, cat_to_name=None, topk=5):
    """
    function which which shows along with the input figure the predicted label plus the top topk
    predicted labels of the model model
    
    Input 
    =====
        image_path(string)
        model(nn.Model) 
        cat_to_name(dict): keys the directories and values the labels
        device: torch device on which calculations are done
    
    Params
    ======
        topk(int) number of top topk probabilities which are shown 
    """
    fig, (ax, ay) = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
    # get the predictions of the model
    predictions = predict(image_path, model, device, topk=topk)
    # invert the class_to_idx dictionary
    idx_to_class = dict([(v, k) for (k, v) in model.class_to_idx.items()])
    # apply the cat_to_name dictionary to obtain the names for the index labels
    names = predictions[1]
    if cat_to_name != None:
        names = np.vectorize(cat_to_name.get)(names)
    
    ay.barh([i for i in range(topk, 0, -1)], predictions[0])
    ay.set_yticks([i for i in range(topk, 0, -1)])
    ay.set_yticklabels(names)
    with Image.open(image_path) as im:
        ax = imshow(process_image(im), ax=ax, title=names[0])
    ay.set_aspect(1./topk)
    plt.show()


