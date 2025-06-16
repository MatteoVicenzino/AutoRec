#HERE WE DEFINE THE MODELS AND THE TRAINING FUNTIONS
import pandas as pd
import numpy as np
import tqdm
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
            nn.Linear(k,1000),
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
            nn.Linear(1000,k),
            nn.Unflatten(1,(2,k))
        )
        self.value_sets = [
            torch.tensor(ids).float(),             # tensore per la prima colonna
            torch.tensor(list(range(0,6))).float() # tensore per la seconda colonna
        ]

    def normalize_columns(self,tensor):
            tensor = tensor.float()
            mean = tensor.mean(dim=0, keepdim=True)  
            std = tensor.std(dim=0, keepdim=True)    
            return (tensor - mean) / std, mean, std

    def denormalize_columns(self,tensor, mean, std):
        tensor = tensor.float()
        return (tensor * std) + mean

    def force_columns_to_values(self, tensor, value_sets):
        tensor = tensor.float()
        output = []
        for col_idx in range(tensor.shape[1]):
            col = tensor[:, col_idx].unsqueeze(1)
            value_set = value_sets[col_idx].unsqueeze(0)  
            distances = torch.abs(col - value_set)
            indices = torch.argmin(distances, dim=1)
            forced_col = value_set.squeeze(0)[indices]
            output.append(forced_col.unsqueeze(1))
        return torch.cat(output, dim=1)



    def forward(self,x):
        device = x.device
        if not hasattr(self, 'value_sets_device'):
            self.value_sets_device = [vs.to(device) for vs in self.value_sets]
        x, mean, std = self.normalize_columns(x)
        z = self.encoder(x)
        recon = self.decoder(z)
        final = self.denormalize_columns(recon, mean, std)
        final = self.force_columns_to_values(final, self.value_sets)
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
        loop = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{N_Epochs}", leave=False)
        for i,us in enumerate(dataloader):
            print(type(us))
            us = us.to(device)
            recon,z = model(us)
            loss = criterion(recon,us)
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

import torch
import torch.nn as nn

#This model is beeing done with AI support to understand wich are the correct dimension during the process
class StringAE(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_dim, pair_emb_dim, hidden_dim, k, string_len):
        super(StringAE, self).__init__()
        self.k = k
        self.string_len = string_len
        self.vocab_size = vocab_size

        # string embedding
        self.string_embedding = nn.Embedding(vocab_size, emb_dim)
        self.encoder_str = nn.LSTM(emb_dim, emb_dim, batch_first=True)

        # number embedding (correzione: interi nelle dimensioni)
        self.encoder_num = nn.Sequential(
            nn.Linear(1, num_dim // 2),
            nn.ReLU(),
            nn.Linear(num_dim // 2, num_dim)
        )
        
        # embedding the combination
        self.pair_encoder = nn.Linear(num_dim + emb_dim, pair_emb_dim)
        
        # encoder sequence con conv1d
        self.encoder_seq = nn.Sequential(
            nn.Conv1d(in_channels=pair_emb_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # decoding combination con ConvTranspose1d
        self.decoder_seq = nn.Sequential(
            nn.ConvTranspose1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=hidden_dim, out_channels=pair_emb_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # decoding numbers
        self.decoder_num = nn.Sequential(
            nn.Linear(pair_emb_dim, num_dim),
            nn.ReLU(),
            nn.Linear(num_dim, num_dim // 2),
            nn.ReLU(),
            nn.Linear(num_dim // 2, 1)
        )

        # decoding strings
        self.decoder_str_lstm = nn.LSTM(pair_emb_dim, hidden_dim, batch_first=True)
        self.decoder_str_fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, num_inputs, str_inputs):
        B, K, L = str_inputs.shape

        # string processing
        str_inputs_flat = str_inputs.view(B * K, L)
        emb_str = self.string_embedding(str_inputs_flat)  
        _, (h_str, _) = self.encoder_str(emb_str)        
        h_str = h_str[-1]                                 
    

        # number processing
        num_inputs_flat = num_inputs.view(B * K, 1)
        h_num = self.encoder_num(num_inputs_flat)         

        # pair processing
        pair = torch.cat([h_num, h_str], dim=1)           
        pair = self.pair_encoder(pair)                     
        pair = pair.view(B, K, -1)                         

        # sequence coding conv1d expects (B, C, L)
        pair = pair.permute(0, 2, 1)                       
        z = self.encoder_seq(pair)                          
        z = z.permute(0, 2, 1)                             

        # decoding sequence convtranspose1d
        dec_seq = z.permute(0, 2, 1)                       
        dec_seq = self.decoder_seq(dec_seq)                
        dec_seq = dec_seq.permute(0, 2, 1)                

        # decoding number
        out_num = self.decoder_num(dec_seq)                
        dec_str_lstm_out, _ = self.decoder_str_lstm(dec_seq)  
        out_str = self.decoder_str_fc(dec_str_lstm_out)        
        out_str = out_str.unsqueeze(2).repeat(1, 1, self.string_len, 1) 

        return out_num, out_str

def train_StringAE(model, dataloader, optimizer, criterion_num, criterion_str, device, N_epochs):
    model.to(device)
    losses = []

    for epoch in range(N_epochs):
        model.train()
        running_loss = 0

        for batch in dataloader:

            batch = batch.to(device)

            num_inputs = batch[..., 1]  
            str_inputs = batch[..., 0]  
            optimizer.zero_grad()
            out_num, out_str = model(str_inputs, num_inputs)  

            # Loss numeri
            loss_num = criterion_num(out_num.squeeze(-1), num_inputs.float())

            # Loss stringhe: usa CrossEntropy su dimensione vocab_size
            B, K, L, V = out_str.shape
            out_str_reshaped = out_str.view(B * K * L, V)               
            target_str = str_inputs.view(B * K * L).long()              

            loss_str = criterion_str(out_str_reshaped, target_str)

            loss = loss_num + loss_str
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{N_epochs} - Loss: {avg_loss:.4f}")

    return losses


        
# nel caso non riuscisse e dovessimo tornare all'imparare le relazioni spaziali questi 2 modelli sono buoni
class C_AE(nn.Module):
    def __init__(self,k):
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
            nn.Sigmoid())
    
    def forward(self,x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon,z

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
            nn.Sigmoid(),
            nn.Unflatten(1,(2,k))
        )
    def forward(self,x):
        z = self.encoder(x)
        final = self.decoder(z)
        return final,z