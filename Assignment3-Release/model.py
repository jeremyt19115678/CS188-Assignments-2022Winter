import torchvision.models as models

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import cv2

class Resnet(nn.Module):
    def __init__(self, mode='finetune'):
        super().__init__()
        """
        use the resnet18 model from torchvision models. Remember to set pretrained as true
        
        mode has three options:
        1) features: to extract features only, we do not want the last fully connected layer of 
            resnet18. Use nn.Identity() to replace this layer.
        2) linear: For this model, we want to freeze resnet18 features, then train a linear 
            classifier which takes the features before FC (again we do not want 
            resnet18 FC). And then write our own FC layer: which takes in the features and 
            output scores of size 40 (because we have 40 categories).
            Because we want to freeze resnet18 features, we have to iterate through parameters()
            of our model, and manually set some parameters to requires_grad = False
            Or use other methods to freeze the features
        3) finetune: Same as 2), except that we we do not need to freeze the features and
           can finetune on the pretrained resnet model.
        """
    ########################################Your Code#################################### 

    #####################################################################################

    def forward(self, x):
    ########################################Your Code#################################### 

    #####################################################################################
        return x

def compute_distances_no_loops(x_train, x_test):
  """
  Copy your implementation from Assignment 1.
  """
  ##########################Your Code###########################################

  ##############################################################################
  #                             END OF YOUR CODE                               #
  ##############################################################################
  return dists

def predict_labels(dists, y_train, k=1):
  """
  Copy your implementation from Assignment 1
  """
  #############################Your Code######################################

  ############################################################################
  return indices, y_pred

class KnnClassifier:
  """
  Copy your implementation from Assignment 1
  """
  #############################Your Code######################################

  ############################################################################

def CAM(feature_conv, weight_softmax, class_idx):
    """
    Implement CAM here
    generate the class activation maps upsample to 256x256
    refer to: https://github.com/zhoubolei/CAM
    """
    ###########################Your Code#######################################

    #############################################################################
    return output_cam
    