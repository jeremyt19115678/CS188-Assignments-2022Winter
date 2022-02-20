import os
import random
from xml.dom import INVALID_ACCESS_ERR
import numpy as np
from PIL import Image
import torch
import cv2
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

def extract_color_features(input):
    input *= 256
    chans = cv2.split(input.numpy().transpose(1,2,0))
    features = []
    colors = ('b', 'g', 'r')
    for (chan, color) in zip(chans, colors):
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        elem = np.max(hist)
        features.append(hist/elem)
    features = torch.tensor(features)
    return features

class Actions(Dataset):
    def __init__(self, split='train', root_dir='', color_feature=False):
        super(Actions, self).__init__()
        
        assert split in ['train', 'test']
        self.split = split

        self.root_dir = root_dir 

        action_file = open(os.path.join(root_dir, "ImageSplits", "actions.txt"))
        all_lines = action_file.readlines()[1:]

        actions = []
        for line in all_lines:
            action = line.strip().split()[0]
            actions.append(action)

        images = []
        labels = []

        for (i,action) in enumerate(actions):
            image_txt = open(os.path.join(root_dir, "ImageSplits", "%s_%s.txt"%(action, split)))
            image_lines = image_txt.readlines()
            for line in image_lines:
                images.append(line.strip())
                labels.append(i)

        self.images = np.array(images)
        self.labels = np.array(labels)
        indices = np.arange(len(self.images))
        np.random.shuffle(indices)
        self.images = self.images[indices]
        self.labels = self.labels[indices]
        self.color_feature = color_feature
        
        if not color_feature:
            self.img_transform = transforms.Compose([ 
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),])
        else:
            self.img_transform = transforms.Compose([ 
                transforms.ToTensor()])

    def __getitem__(self, index):
        file = self.images[index]
        label = self.labels[index]
        path = os.path.join(self.root_dir, "actions", file)
        image = Image.open(path).convert("RGB")

        image = self.img_transform(image)

        if self.color_feature:
            image = extract_color_features(image)

        return image, label
    
    def __len__(self):
        return len(self.images)
