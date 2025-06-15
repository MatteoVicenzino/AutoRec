#HERE WE DEFINE THE MODELS AND THE TRAINING FUNTIONS
import pandas as pd
import numpy as np

import math
from time import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision 
import torch.utils.data as data
import torch.distributions as dist

<<<<<<< HEAD
#model definition
class F_AE(nn.Module):
    def __init__(self,k, h, ids):
        super(F_AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(k,1000),
=======

class F_AE(nn.Module):
    def __init__(self,k, ids):
        super(F_AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(k,100),
>>>>>>> origin/main
            nn.ReLU(),
            nn.Linear(1000,500),
            nn.ReLU(),
            nn.Linear(500,250),
            nn.ReLU(),
            nn.Linear(250,125),
            nn.ReLU(),
            nn.Linear(125,75),
            nn.ReLU(),
            nn.Linear(75,50)
        )
        self.decoder = nn.Sequential(
            nn.Linear(50,75),
            nn.ReLU(),
            nn.Linear(75,125),
            nn.ReLU(),
            nn.Linear(125,250),
            nn.ReLU(),
            nn.Linear(250,500),
            nn.ReLU(),
<<<<<<< HEAD
            nn.Linear(500,1000),
            nn.ReLU(),
            nn.Linear(1000,k),
=======
            nn.Linear(100,k),
>>>>>>> origin/main
            nn.Unflatten(1,(2,k))
        )
        self.value_sets = [ids,list(range(0,6))]

    def normalize_columns(self,tensor):
<<<<<<< HEAD
            #This function normilize the 2 colimns book_id and rating separatly. this is done to mantain a clear difference between the two parts
=======
>>>>>>> origin/main
            tensor = tensor.float()
            mean = tensor.mean(dim=0, keepdim=True)  
            std = tensor.std(dim=0, keepdim=True)    
            return (tensor - mean) / std, mean, std

    def denormalize_columns(self,tensor, mean, std):
<<<<<<< HEAD
        # with this function we return from the normalization
=======
>>>>>>> origin/main
        tensor = tensor.float()
        return (tensor * std) + mean

    def force_columns_to_values(self,tensor, value_sets):
<<<<<<< HEAD
        #this function is needed to remap the columns in the correct domains
=======
>>>>>>> origin/main
        tensor = tensor.float()
        output = []
        for col_idx in range(tensor.shape[1]):
            col = tensor[:, col_idx].unsqueeze(1)  
            value_set = torch.tensor(value_sets[col_idx]).float().unsqueeze(0)  
            distances = torch.abs(col - value_set)  
            indices = torch.argmin(distances, dim=1)
            forced_col = value_set.squeeze(0)[indices] 
            output.append(forced_col.unsqueeze(1))  
        return torch.cat(output, dim=1)  


    def forward(self,x):
        x, mean, std = self.normalize_columns(x)
        z = self.encoder(x)
        recon = self.decoder(z)
        final = self.force_values(recon)
        final = self.denormalize_columns(final, mean, std)
        final = self.force_columns_to_values(final, self.value_sets)
        return final,z
    

def train(model, N_Epochs, dataloader, criterion, optimazer):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    losses = []
    start = time()

    for epoch in range(N_Epochs):
        model.train()
        Tr_current_loss = 0

        for i,us in enumerate(dataloader):
            us = us.to(device)

            recon,z = model(us)
            loss = criterion(recon,us)
            optimazer.zero_grad()
            loss.backward()
            optimazer.step()
            Tr_current_loss += loss.item()
            
        losses.append(Tr_current_loss/i)
        print(f'Epoch: {epoch+1} | Loss: {Tr_current_loss/i:.4f} | Time: {time()-start:.2f}')
        start = time()
    return losses,recon,z

def loss_graph(tr_loss,n_epochs):
    plt.plot(range(n_epochs),tr_loss,label='tr_loss', c='black')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.show()

        
# nel caso non riuscisse e dovessimo tornare all'imparare le relazioni spaziali questi 2 modelli sono buoni
class C_AE(nn.Module):
    def __init__(self,k):
<<<<<<< HEAD

        super(C_AE, self).__init__()
        self.out_len1 = math.floor((k - 20) / 20) + 1
        self.out_len2 = math.floor((self.out_len1 - 10) / 10) + 1
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels = 1, out_channels = 5, kernel_size = 20, stride = 20),
            nn.ReLU(),
            nn.Conv1d(in_channels = 5, out_channels = 10, kernel_size = 10, stride = 10),
            nn.ReLU(),
            nn.flatten(),
            nn.linear(10*self.out_len2,50),
            nn.ReLU(),
            nn.linear(50,25)            
        )
        self.decoder = nn.Sequential(
            nn.linear(25,50),
            nn.ReLU(),
            nn.linear(50,10*self.out_len2),
            nn.Unflatten(1,(10,self.out_len2)),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels = 10, out_channels = 5, kernel_size = 10, stride = 10),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels = 5, out_channels = 1, kernel_size = 20, stride = 20),
            nn.Sigmoid()
=======
        super(C_AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels = 1, out_channels = 5, kernel_size = 20, stride = 20),
            nn.ReLU(),
            
>>>>>>> origin/main
        )

class F_AE2(nn.Module):
    def __init__(self,k):
        super(F_AE2, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(k,100),
            nn.ReLU(),
            nn.Linear(100,50),
            nn.ReLU(),
            nn.Linear(50,25),
            nn.ReLU(),
            nn.Linear(25,10),
            nn.ReLU(),
            nn.Linear(10,1)
        )
        self.decoder = nn.Sequential(
            nn.Linear(1,10),
            nn.ReLU(),
            nn.Linear(10,25),
            nn.ReLU(),
            nn.Linear(25,50),
            nn.ReLU(),
            nn.Linear(50,100),
            nn.ReLU(),
            nn.Linear(100,k),
            nn.Unflatten(1,(2,k))
        )
    def forward(self,x):
        z = self.encoder(x)
        final = self.decoder(z)
        return final,z
