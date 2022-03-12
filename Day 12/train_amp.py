#Import Libraries
import matplotlib.pyplot as plt
from PIL import Image as im
import numpy as np

#Profiling
import nvidia_dlprof_pytorch_nvtx as nvtx
nvtx.init(enable_function_stack=True)

#Automatic Mixed Precision
from apex import amp
#REF: https://medium.com/pytorch/catalyst-101-accelerated-pytorch-bd766a556d92
#Prepare Dataset

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = MNIST("./mnist", train=True, download=True, transform=ToTensor())
valid_dataset = MNIST("./mnist", train=False, download=True, transform=ToTensor())

train_loader = DataLoader(train_dataset, batch_size=32, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=32, num_workers=4)

#REF: https://github.com/amitrajitbose/handwritten-digit-recognition/blob/master/handwritten_digit_recognition_GPU.ipynb
#View details of Dataset
dataiter = iter(train_loader)
images, labels = dataiter.__next__()
print(type(images))
print(images.shape)
print(labels.shape)

#Sample data from the Training Dataset
plt.imshow(images[0].numpy().squeeze(), cmap='gray_r')

# Model

from torch import nn

model = nn.Sequential(
    nn.Linear(28 * 28, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)
model = model.to(device)
print(model)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#AMP
model, optimizer = amp.initialize(model, optimizer, opt_level="O1", loss_scale="dynamic")

# mnist criterion
criterion = nn.CrossEntropyLoss()


# PYTORCH training loop
with torch.autograd.profiler.emit_nvtx():
  num_epochs = 1
  for epoch in range(num_epochs):

      # train
      for x,y in train_loader:
          x = x.to(device)
          y = y.to(device)
          x = x.view(len(x), -1)

          logits = model(x)
          loss = criterion(logits, y)
     
          print("train loss: ", loss.item())
          #AMP
          with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

          optimizer.step()
          optimizer.zero_grad()

      # validation
      with torch.no_grad():
          valid_loss = []
          for x,y in valid_loader:
              x = x.to(device)
              y = y.to(device)
              x = x.view(len(x), -1)
              logits = model(x)
              valid_loss.append(criterion(logits, y).item())

          valid_loss = torch.mean(torch.tensor(valid_loss))
          print("valid loss: ", valid_loss.item())
