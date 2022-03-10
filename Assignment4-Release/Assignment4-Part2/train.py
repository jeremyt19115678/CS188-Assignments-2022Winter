import os
import time
import random

from model import *
from dataset import *
import torch
import torch.nn as nn
from tqdm import tqdm

def group_weight(module):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups
    
def pixel_acc(pred, label):
    _, preds = torch.max(pred, dim=1)
    valid = (label >= 0).long()

    acc_sum = torch.sum(valid * (preds == label).long())
    pixel_sum = torch.sum(valid)

    acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
    return acc

def train_seg(train_dataloader, loss_criterion, encoder, decoder, optimizer1, optimizer2):
    total_loss = 0.0
    total_acc = 0.0

    encoder.train()
    decoder.train()

    for i, data in enumerate(tqdm((train_dataloader))):
        data['img_data'], data['seg_label'] = data['img_data'].cuda(), data['seg_label'].squeeze().cuda()

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        pred = decoder(encoder(data['img_data']))
        loss = loss_criterion(pred, data['seg_label'])
        acc = pixel_acc(pred, data['seg_label'])

        loss = loss.mean()
        acc = acc.mean()

        total_loss += loss.item()
        total_acc += acc.item()

        loss.backward()
        optimizer1.step()
        optimizer2.step()

    total_loss /= i
    total_acc /= i

    return total_loss, total_acc

def val_seg(val_dataloader, loss_criterion, encoder, decoder):
    total_loss = 0.0
    total_acc = 0.0

    encoder.eval()
    decoder.eval()

    for i, data in enumerate(tqdm((val_dataloader))):
        data['img_data'], data['seg_label'] = data['img_data'].cuda(), data['seg_label'].squeeze().cuda()

        with torch.no_grad():
            pred = decoder(encoder(data['img_data']))

            loss = loss_criterion(pred, data['seg_label'])
            acc = pixel_acc(pred, data['seg_label'])
            loss = loss.mean()
            acc = acc.mean()

            total_loss += loss.item()
            total_acc += acc.item()

    total_loss /= i
    total_acc /= i

    return total_loss, total_acc
