from definitions import *
from tau_optim import *
from train import *

import numpy as np
import numpy.linalg as npla
import scipy as sp
import matplotlib.pyplot as plt

import torch, torchvision, torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from  torchvision.datasets import MNIST

from torchvision.models import resnet18
from torch import nn, optim

from tqdm.autonotebook import tqdm
import inspect
import time 
from sklearn.metrics import accuracy_score

model = resnet18(num_classes=10, )
model.conv1 =  torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
if torch.cuda.is_available():
    model.cuda()

model.load_state_dict(torch.load('/content/weights_additive_loss.pth'))


transform1 = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,)), transforms.RandomAffine(degrees=(0, 360), translate=(0.1, 0.3), scale=(0.70, 0.75))])
testset_aff = MNIST('/test_aff1', train=False, download=True, transform = transform1)
test_loader_aff = DataLoader(testset_aff, batch_size=64, shuffle=True, num_workers=2)

losses = []
val_losses = 0
precision, recall, f1, accuracy = [], [], [], []

model.eval()
val_batches = len(test_loader_aff)
with torch.no_grad():
  for i, data in enumerate(test_loader_aff):
      X, y = data[0].to(device), data[1].to(device)
      # b3.batchSize = data[0].shape[0]
      # X = b3(X,y)
      outputs = model(X) # this get's the prediction from the network
      val_losses += loss_function(outputs, y)
      predicted_classes = torch.max(outputs, 1)[1] # get class from network's prediction
      accuracy.append(
              accuracy_score(y.cpu(), predicted_classes.cpu())
          )
        
  print(f" validation loss: {val_losses/val_batches}")
  print_scores(accuracy, val_batches)