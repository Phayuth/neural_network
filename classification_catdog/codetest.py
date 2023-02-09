import os
os.environ['CUDA_VISIBLE_DEVICES'] ='0'

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from dataset_import import dataset_import
from model import Model

import matplotlib.pyplot as plt

dataset_train = dataset_import('./dataset/train/')
dataset_test  = dataset_import('./dataset/test/')


print(dataset_train.data_len)
print(dataset_train.image_list[0])
print(dataset_test.data_len)
print(dataset_test.image_list[0])

imt, label = dataset_train.__getitem__(7520)
print(imt.size())
print(label)
plt.imshow(imt)


# fake data
f_data = torch.rand((8,100,100)).to("cuda")
outp = Model(f_data)
print(f_data.shape)
print(outp.shape)