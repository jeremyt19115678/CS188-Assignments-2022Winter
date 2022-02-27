import torchvision.models as models

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import cv2
import math
import sys
import os
try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding, kernel size 3, padding 1 and no bias"
    #######################Your Code Here#################################

    ######################################################################

class Resnet(nn.Module):
    def __init__(self, mode='finetune'):
        super().__init__()
        """Define self.resnet as pretrained resnet18 from torchvision models"""
        #####################################Your Code###############################

        #############################################################################

    def forward(self, x):
        """Return conv_out, which is a python list of the features output by layer1, """
        """layer2, layer3, layer4 of resnet"""
        """hint: https://github.com/CSAILVision/semantic-segmentation-pytorch/blob/master/mit_semseg/models/models.py"""

        conv_out = []

        ####################################Your Code#################################

        #############################################################################
        return conv_out

def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    "3x3 convolution + BN + relu, kernel size 3, padding 1, no bias"
    ######################################Your Code####################################

    ###################################################################################

class UPerNet(nn.Module):
    """Implement the UPerNet from https://github.com/CSAILVision/semantic-segmentation-pytorch/blob/master/mit_semseg/models/models.py. You can just copy the codes and modify it."""
    ####################################################################Your Code#########################################################

    #############################################################################################################3