from definitions import *

import numpy as np
import numpy.linalg as npla
import scipy as sp
import matplotlib.pyplot as plt

import torch, torchvision, torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from  torchvision.datasets import MNIST


def rotation_mat(theta):
    
    sin = np.sin(theta)
    cos = np.cos(theta)
    mat = np.array([[cos, sin], [-sin, cos]])
    return mat

#Loading the dataset
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])

trainset = MNIST('/train', train=True, download=True, transform = transform)
testset = MNIST('/test', train=False, download=True, transform = transform)
train_batch_size = 64
test_batch_size = 64
val_batch_size = 64
valset, lite  = random_split(MNIST('/val', train=False, download=True, transform = transform), [6000, 4000])
print(len(valset), len(lite))
trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=64, shuffle=True, num_workers=2)
valloader = DataLoader( valset , batch_size = val_batch_size, shuffle=False)


uncomment 
dataiter = iter(trainloader)
images, labels = next(dataiter)
print('Labels: ', labels)
print('Batch shape: ', images.size())
image = images[0][0]
plt.imshow(image)
plt.show()

im = images[0].permute(1,2,0).numpy()
# im = images[0].numpy()
print(im.shape)
plt.imshow(im[:,:,0])
# plt.imshow(im[0,:,:])

A = rotation_mat(np.deg2rad(30))
b = np.zeros((2,))
im1 = im.copy()
print(im1.shape)
# print(A)
tau0, tau1 = affine_to_vf2(A,b,28,28)
im2 = image_interpolation_bicubic2(im1, tau0, tau1)

fig = plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.imshow(im[:,:,0])
plt.subplot(1,2,2)
plt.imshow(im2[:,:,0])
plt.show()