
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
class FAE(nn.Module):
    def __init__(self,k, ids):
        super(FAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(k*2,1000),
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
            nn.Linear(500,1000),
            nn.ReLU(),
            nn.Linear(1000,k*2),
            nn.Unflatten(1,(k,2))
        )
        self.value_sets = [torch.tensor(ids).float(),torch.tensor(list(range(0,6))).float() ]

    def normalize_columns(self, tensor):
        tensor = tensor.float()
        mean = tensor.mean(dim=0, keepdim=True)   
        std = tensor.std(dim=0, keepdim=True) + 1e-8  
        return (tensor - mean) / std, mean, std

    def denormalize_columns(self, tensor, mean, std):
        return tensor * std + mean

    def force_columns_to_values(self, tensor, value_sets):
        tensor = tensor.float()
        output = []
        for i in range(tensor.shape[2]):  
            col = tensor[:, :, i]          
            value_set = value_sets[i].to(tensor.device)  
            col = col.unsqueeze(1)        
            value_set = value_set.unsqueeze(0).unsqueeze(-1)  
            distances = torch.abs(col - value_set)           
            indices = torch.argmin(distances, dim=1)        
            forced_col = value_set.squeeze(0)[indices]    
            forced_col = forced_col.squeeze(-1)         
            output.append(forced_col)                      
        return torch.stack(output, dim=2)                  




    def forward(self,x, force_values=False):
        device = x.device
        if not hasattr(self, 'value_sets_device'):
            self.value_sets_device = [vs.to(device) for vs in self.value_sets]
        x, mean, std = self.normalize_columns(x)
        z = self.encoder(x)
        recon = self.decoder(z)
        final = self.denormalize_columns(recon, mean, std)
        if force_values:
            final = self.force_columns_to_values(final, self.value_sets_device)
        return final,z
    

def train_FAE(model, N_Epochs, dataloader, criterion, optimizer, scheduler = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    losses = []
    parameters = []
    start = time()
    best_loss = float('inf')

    for epoch in range(N_Epochs):
        model.train()
        Tr_current_loss = 0
        for i,us in enumerate(dataloader):
            us = us.to(device)
            recon,z = model(us)
            loss = criterion(model.value_sets,us, recon)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            Tr_current_loss += loss.item()
            
        losses.append(Tr_current_loss/i)
        print(f'Epoch: {epoch+1} | Loss: {Tr_current_loss/i:.4f} | Time: {time()-start:.2f}')
        parameters.append(model.state_dict())
        if best_loss > Tr_current_loss:
            best_loss = Tr_current_loss
            torch.save(model.state_dict(), 'best_model.pth')
        
        if scheduler is not None:
            scheduler.step()

        start = time()
    return losses,parameters,recon,z

def loss_graph(tr_loss,n_epochs):
    plt.plot(range(n_epochs),tr_loss,label='tr_loss', c='black')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.show()


def string_to_tensor(stringhe, s, max_len):
    vocab = sorted(list(set("".join(stringhe))))  # tutti i caratteri unici
    char2idx = {ch: i for i, ch in enumerate(vocab)}
    idx2char = {i: ch for ch, i in char2idx.items()}
    vocab_size = len(vocab)
    def string_to_indices(s, max_len):
        indices = [char2idx[c] for c in s]
        return indices + [0] * (max_len - len(indices))

    max_len = max(len(s) for s in stringhe)
    X = torch.tensor([string_to_indices(s, max_len) for s in stringhe])

        
# nel caso non riuscisse e dovessimo tornare all'imparare le relazioni spaziali questi 2 modelli sono buoni
class Spatial_C_AE(nn.Module):
    def __init__(self,k):
        super(Spatial_C_AE, self).__init__()
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
            nn.Sigmoid())
    
    def forward(self,x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon

class Spatial_F_AE(nn.Module):
    def __init__(self,k):
        super(Spatial_F_AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(k,500),
            nn.ReLU(),
            nn.Linear(500,250),
            nn.ReLU(),
            nn.Linear(250,125),
            nn.ReLU(),
            nn.Linear(125,50),
            nn.ReLU(),
            nn.Linear(50,25)
        )
        self.decoder = nn.Sequential(
            nn.Linear(25,50),
            nn.ReLU(),
            nn.Linear(50,125),
            nn.ReLU(),
            nn.Linear(125,250),
            nn.ReLU(),
            nn.Linear(250,500),
            nn.ReLU(),
            nn.Linear(500,k),
            nn.Sigmoid(),
        )
    def forward(self,x):
        z = self.encoder(x)
        final = self.decoder(z)
        return final
    
class Spatial_LSTM_AE(nn.Module):
    def __init__(self,input_size=1, hidden_size=64, latent_size=32, num_layers=10):
        super(Spatial_LSTM_AE, self).__init__()
        self.encoder_lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc_enc = nn.Linear(hidden_size, latent_size)
        self.fc_dec = nn.Linear(latent_size, hidden_size)
        self.decoder_lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, input_size)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        _, (hn, _) = self.encoder_lstm(x)
        h_last = hn[-1]
        z = self.fc_enc(h_last)
        h_dec = self.fc_dec(z)
        h_dec = h_dec.unsqueeze(0).repeat(self.decoder_lstm.num_layers, 1, 1)
        c_dec = self.fc_dec(z)       # fc_c è un altro Linear
        c_dec = c_dec.unsqueeze(0).repeat(self.decoder_lstm.num_layers, 1, 1)
        decoder_input = torch.zeros(batch_size, seq_len, 1).to(x.device)
        decoder_output, _ = self.decoder_lstm(decoder_input, (h_dec, c_dec))
        reconstructed = self.output_layer(decoder_output)  
        return reconstructed

def train_spatial(model, dataloader, criterion, optimizer, num_epochs, best_loss=float('inf')):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    losses = []
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, batch in enumerate(dataloader):
            batch = batch.to(device)
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        losses.append(running_loss / (i + 1))
        if running_loss < best_loss:
            best_loss = running_loss
            torch.save(model.state_dict(), 'best_model.pth')
        print(f"Epoch {epoch+1}: Loss = {running_loss / (i + 1):.4f}")
    return losses

    
