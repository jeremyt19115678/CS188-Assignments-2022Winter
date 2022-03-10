import os
import json
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2

def imresize(im, size, interp='bilinear'):
    if interp == 'nearest':
        resample = Image.NEAREST
    elif interp == 'bilinear':
        resample = Image.BILINEAR
    elif interp == 'bicubic':
        resample = Image.BICUBIC
    else:
        raise Exception('resample method undefined!')

    return cv2.resize(im, size, resample)

def round2nearest_multiple(x, p):
    return ((x - 1) // p + 1) * p

class ADEDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, split):
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        
        self.root_dir = root_dir
        self.split = split

        self.files = os.listdir(os.path.join(self.root_dir, "images", self.split))

    def img_transform(self, img):
        # 0-255 to 0-1
        img = np.float32(np.array(img)) / 255.
        img = img.transpose((2, 0, 1))
        img = self.normalize(torch.from_numpy(img.copy()))
        return img

    def segm_transform(self, segm):
        # to tensor, -1 to 149
        segm = torch.from_numpy(np.array(segm)).long() - 1
        return segm

    def __getitem__(self, index):

        image_path = os.path.join(self.root_dir, "images", self.split, self.files[index])
        segm_path = os.path.join(self.root_dir, "annotations", self.split, self.files[index].replace("jpg", "png"))

        img = Image.open(image_path).convert('RGB')
        segm = Image.open(segm_path)

        assert(segm.mode == "L")
        assert(img.size[0] == segm.size[0])
        assert(img.size[1] == segm.size[1])

        ori_width, ori_height = img.size

        img = self.img_transform(img)

        segm_rounded_width = round2nearest_multiple(segm.size[0], 4)
        segm_rounded_height = round2nearest_multiple(segm.size[1], 4)
        segm_rounded = Image.new('L', (segm_rounded_width, segm_rounded_height), 0)
        segm_rounded.paste(segm, (0, 0))
        segm_rounded = np.array(segm_rounded)
        segm = imresize(
            segm_rounded,
            (segm_rounded.shape[0] // 4, \
                segm_rounded.shape[1] // 4), \
            interp='nearest')

        # segm transform, to torch long tensor HxW
        segm = self.segm_transform(segm)
        batch_segms = torch.unsqueeze(segm, 0)

        output = dict()
        output['img_data'] = np.array(img)
        output['seg_label'] = batch_segms.contiguous()
        return output

    def __len__(self):
        return len(self.files)





