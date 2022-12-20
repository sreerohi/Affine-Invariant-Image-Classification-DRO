from definitions import *
from tau_optim import *

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
from sklearn.metrics import  accuracy_score


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


model = resnet18(num_classes=10, )
model.conv1 =  torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
if torch.cuda.is_available():
    model.cuda()


def initialize_weights(m):
  if isinstance(m, nn.Conv2d):
      # print("conv2d")
      nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
      if m.bias is not None:
          nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, nn.BatchNorm2d):
      # print("batch norm")
      nn.init.constant_(m.weight.data, 1)
      nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, nn.Linear):
      # print("linear")
      nn.init.kaiming_uniform_(m.weight.data)
      nn.init.constant_(m.bias.data, 0)

model.apply(initialize_weights)


def print_scores(a=None, batch_size=None):
    for name, scores in zip(("accuracy"), (a)):
        print(f"\t{name.rjust(14, ' ')}: {sum(scores)/batch_size:.4f}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
loss_function = nn.CrossEntropyLoss() # your loss function, cross entropy works well for multi-class problems

optimizer = optim.Adadelta(model1.parameters())



epochs_model = 5
train_loader, val_loader = trainloader, valloader

loss_function = nn.CrossEntropyLoss()

optimizer = optim.Adadelta(model.parameters())

start_ts = time.time()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

losses = []
batches = len(train_loader)
val_batches = len(val_loader)
# bI = BatchInterpolOptim(batchSize=train_batch_size, M = 28, N = 28, loss_fn = loss_function, step_size = 0.001, model = model, nOptSteps = 1, device = device)

b3 = None
b3 = BatchInterpolOptim(64, 28, 28, loss_function, 10, model, device)
epochs_tau = 2

for epoch in range(epochs_model):
    total_loss = 0

    #Optimizing Taus:
    model.eval()        
    lossValues = []
    for e in range(epochs_tau):
        timeEpoch = time.time()
        total_loss = 0
        progress1 = tqdm(enumerate(train_loader), desc="Tau %complete: ", total=batches)
        for i, data in progress1:
            X, y = data[0].to(device), data[1].to(device)
            b3.batchSize = data[0].shape[0]
            tausi = b3.taus
            b3(X, y)
            b3.computegradLoss_taus(X)
            b3.optimize()
            lossValues.append(b3.lossEpoch)
        print(f"Training Taus time for 1 epoch: {time.time()-timeEpoch}s")
    plt.plot(np.arange(0, len(lossValues)), lossValues)
    plt.title("loss, optimizing tau")
    plt.show()

    progress = tqdm(enumerate(train_loader), desc="Loss: ", total=batches)
    model.train()
    loss_model_per_epoch = []    
    for i, data in progress:
      # X, y = data[0].to(device), data[1].to(device)
      X, y = data[0].to(device), data[1].to(device)
      b3.batchSize = data[0].shape[0]
      X2 = b3(X,y)

      model.zero_grad()
      outputs1 = model(X)
      outputs2 = model(X2)
      # print(outputs.shape)
      loss = loss_function(outputs1, y) + loss_function(outputs2, y)
      
      loss.backward()

      optimizer.step()

      # getting training quality data
      current_loss = loss.item()
      total_loss += current_loss
      loss_model_per_epoch.append(total_loss/(i+1))
      # updating progress bar
      progress.set_description("Loss: {:.4f}".format(total_loss/(i+1)))

    plt.plot(np.arange(len(loss_model_per_epoch)), loss_model_per_epoch)
    plt.title("loss of model per epoch")
    plt.show()   

    if torch.cuda.is_available():
      torch.cuda.empty_cache()
    val_losses = 0
    precision, recall, f1, accuracy = [], [], [], []

    model.eval()
    with torch.no_grad():
      for i, data in enumerate(val_loader):
          X, y = data[0].to(device), data[1].to(device)

          outputs = model(X) # this get's the prediction from the network

          val_losses += loss_function(outputs, y)

          predicted_classes = torch.max(outputs, 1)[1] # get class from network's prediction

          accuracy.append(
                  accuracy_score(y.cpu(), predicted_classes.cpu())
              )
        
    print(f"Epoch {epochs_model+1}/{epochs_model}, training loss: {total_loss/batches}, validation loss: {val_losses/val_batches}")
    print_scores(accuracy, val_batches)
    losses.append(total_loss/batches)
    print(f"Training time: {time.time()-start_ts}s")

torch.save(model.state_dict(), 'weights_additive_loss.pth')