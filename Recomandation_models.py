#HERE WE DEFINE THE MODELS AND THE TRAINING FUNTIONS
import pandas as pd
import numpy as np
from time import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision 
import torch.utils.data as data
import torch.distributions as dist

def force_values(tensor):
    value_set = [0,1,2,3,4,5]
    min_val = min(value_set)
    max_val = max(value_set)
    tensor = torch.clamp(tensor, min_val, max_val)
    tensor = tensor.unsqueeze(-1)  
    value_set_tensor = torch.tensor(value_set, device=tensor.device).unsqueeze(0)  
    distances = torch.abs(tensor - value_set_tensor)  
    closest_indices = torch.argmin(distances, dim=-1) 
    result = value_set_tensor.gather(-1, closest_indices.unsqueeze(-1)).squeeze(-1)
    return result

class F_AE(nn.Module):
    def __init__(self,k):
        super(F_AE, self).__init__()
        self.encoder = nn.Sequential(
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
            nn.Linear(100,k)

        )
        
    def forward(self,x):
        z = self.encoder(x)
        recon = self.decoder(z)
        final = force_values(recon)
        return final,x
    
class C_AE(nn.Module):
    def __init__(self,k):
        super(C_AE, self).__init__()
        self.encoder = nn.Sequential(
        nn.Conv1d()
        )

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

def Vae_graphs(tr_loss,n_epochs):
    plt.plot(range(n_epochs),tr_loss,label='tr_loss', c='black')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.show()

        