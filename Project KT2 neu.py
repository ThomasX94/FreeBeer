# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 21:46:40 2020

@author: User
"""
#%%
import numpy as np
import torch
import torch.nn as nn
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import random
import time

#%% Download question Files
quest = pd.read_csv("questions.csv")
print(len(quest))
n_quest=18144
n_part=8
n_responses=3

#%% Download User Files
users = []
for i in range(1, 5):
    u = pd.read_csv("Subset KT1/u"+str(i)+".csv")
    users.append(u)

#users = pd.DataFrame(users)
            
#%% Add Part and Correct Answer
def add_part_and_correct_answer(users):
    for u in users:
        for index, row in u.iterrows():
            question_id=row["question_id"]
            part=quest.loc[quest["question_id"]==question_id, "part"].values
            u.at[index, "part"] = part-1
            correct_answer = quest.loc[quest["question_id"] == question_id, "correct_answer"].values
            if (row["user_answer"] == correct_answer):
                u.at[index, "answered_correct"] = 1
            else:
                u.at[index, "answered_correct"] = 0
            
add_part_and_correct_answer(users)

#%% Convert Question ID to an integer
def modify_questionid(users):
    for u in users:
        for index, row in u.iterrows():
            question_id = row["question_id"]
            question_id = question_id.replace('q', '')
            question_id = int(question_id)
            u.at[index, "question_id"] = question_id
            
modify_questionid(users)
#%% Future Mask
def future_mask(seq_len):
    future_mask=np.triu(np.ones((seq_len,seq_len)), k=1).astype("bool")
    return torch.from_numpy(future_mask)

#%% 
MAX_SEQ = 50
class SAINTDataset(Dataset):
    def __init__(self, users, max_seq = MAX_SEQ):
        self.max_seq = max_seq
        self.samples = users
        
    def __getitem__(self, index):
        user= self.samples[index]
        column_user=list(user)
        q = pd.DataFrame(np.zeros((self.max_seq, 7)), dtype=str, columns=column_user)
        response = pd.DataFrame(np.zeros((self.max_seq)))

        seq_len = len(user)
        
        if (seq_len >= self.max_seq):
            q[:] = user[:self.max_seq].values
        else:
            q[-seq_len:]=user[:].values
            
        part = q["part"]
        question_id = q["question_id"]
        y = q["answered_correct"]
        
        
        if (seq_len >= self.max_seq):
            response.loc[0] = 2
            response[1:] = y[:self.max_seq-1].values
        else:
            token_index = self.max_seq - seq_len
            response[1:] = y[:self.max_seq-1].values
            response.loc[token_index] = 2
            
        response = response.apply(pd.to_numeric).astype('int64')
        response = torch.from_numpy(response.values)
        response = torch.squeeze(response, 1)
         
        part = part.apply(pd.to_numeric).astype('int64')
        part = torch.from_numpy(part.values)
        
        question_id = question_id.apply(pd.to_numeric).astype('int64')
        question_id = torch.from_numpy(question_id.values)
        
        y = y.apply(pd.to_numeric).astype('int64')
        y = torch.from_numpy(y.values)
        #y = torch.unsqueeze(y,1)
        
        return part, question_id, y, response
        
        
    def __len__(self):
        return len(users)
        
#%%
train, val = train_test_split(users, test_size=0.2)

train_dataset = SAINTDataset(train)

train_dataloader= DataLoader(train_dataset, 
                             batch_size=1
                             )

val_dataset = SAINTDataset(val)

print(train_dataset[1])
    
#%%     
class FFN(nn.Module):      
    def __init__(self, state_size):
        super(FFN, self).__init__()
        self.state_size = state_size
        self.l1=nn.Linear(state_size, state_size)
        self.relu=nn.ReLU()
        self.l2=nn.Linear(state_size, state_size)
        
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out
     

#%%
class EncoderBlock(nn.Module):
    def __init__(self, embed_dim=64):
        super(EncoderBlock, self).__init__()
        self.mha_encoder=nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8)
        self.ffn= FFN(embed_dim)
        self.layer_norm1=nn.LayerNorm(embed_dim)
        self.layer_norm2=nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        out = x.permute(1,0,2)
        seq_len = out.shape[0]
        skipout = out
        out, attn_weights = self.mha_encoder(out, out, out, attn_mask=future_mask(seq_len=seq_len))                           
        out=skipout+out
        out = self.layer_norm1(out) 
        skipout=out
        out=self.ffn(out)
        out=skipout+out
        out=self.layer_norm2(out)
        out = out.permute(1,0,2)
        return out

#%%
class Encoder(nn.Module):
    def __init__(self, embed_dim=64):
        super(Encoder, self).__init__()
        self.encoder_block1 = EncoderBlock()
        self.encoder_block2 = EncoderBlock()
        self.encoder_block3 = EncoderBlock()
        self.encoder_block4 = EncoderBlock()
        
    def forward(self, x):
        out = self.encoder_block1(x)
        out = self.encoder_block2(out)
        out = self.encoder_block3(out)
        out = self.encoder_block4(out)
        return out
        
#%%
class DecoderBlock(nn.Module):
    def __init__(self, embed_dim=64):
        super(DecoderBlock, self).__init__()
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.layer_norm3 = nn.LayerNorm(embed_dim)
        self.mha1 = nn.MultiheadAttention(embed_dim=embed_dim, num_heads = 8)
        self.mha2 = nn.MultiheadAttention(embed_dim=embed_dim, num_heads = 8)
        self.ffn = FFN(embed_dim)
        
    def forward(self, x, en_out):
        out = x.permute(1, 0, 2)
        seq_len = out.shape[0]
        att_out, attn_weights = self.mha1(out, out, out, attn_mask = future_mask(seq_len=seq_len))
        out = out+att_out
        out = self.layer_norm1(out)
        att_out, attn_weights = self.mha2(out, en_out, en_out, attn_mask =future_mask(seq_len=seq_len))
        out = out + att_out
        out = self.layer_norm2(out)
        ffn_out = self.ffn(out)
        out =ffn_out + out
        out = self.layer_norm3(out)
        out = out.permute(1, 0, 2)
        return out
        
        
#%% 
class Decoder(nn.Module):
    def __init__(self, embed_dim=64):
        super(Decoder, self).__init__()
        self.decoder_block1 = DecoderBlock()
        self.decoder_block2 = DecoderBlock()
        self.decoder_block3 = DecoderBlock()
        self.decoder_block4 = DecoderBlock()
        
    def forward(self, x, en_out):
        out = self.decoder_block1(x, en_out)
        out = self.decoder_block2(out, en_out)
        out = self.decoder_block3(out, en_out)
        out = self.decoder_block4(out, en_out)
        return out
        

#%%
class SAINT(nn.Module):
    def __init__(self, n_quest, n_part, n_responses, embed_dim=64, max_seq=MAX_SEQ, device = 'cpu'):
        super(SAINT, self).__init__()
        self.device = device
        self.n_quest=n_quest
        self.n_part=n_part
        self.embed_dim = embed_dim
        self.ex_embed = nn.Embedding(n_quest, embed_dim)
        self.part_embed = nn.Embedding(n_part, embed_dim)
        self.pos_embed = nn.Embedding(max_seq, embed_dim)
        self.res_embed = nn.Embedding(n_responses, embed_dim)
        self.encoder= Encoder() 
        self.decoder= Decoder()
        self.output = nn.Linear(in_features=embed_dim, out_features=1)
        self.max_seq=max_seq
        
    def forward(self, part, question_id, response):
        part_emb = self.part_embed(part)
        ex_emb = self.ex_embed(question_id)
        pos_ids = torch.arange(self.max_seq).to(self.device)
        pos_emb = self.pos_embed(pos_ids)
        res_emb = self.res_embed(response)
        
        
        de_in = res_emb + pos_emb
        en_in = ex_emb + part_emb + pos_emb
        en_out = self.encoder(en_in)
        en_out = en_out.permute(1,0,2)
        de_out = self.decoder(de_in, en_out)
        print(de_out.shape)
        output = self.output(de_out)
        return output
#%% Training Model        
device = 'cuda' if torch.cuda.is_available()  else 'cpu'      
model = SAINT(n_quest, n_part, n_responses, device = device)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
lossfunction = nn.CrossEntropyLoss()

model.to(device)
lossfunction.to(device)
num_epochs = 1

#%% Critical Part: Expected Target Size 1,1 got torch.Size(1,50) 
def train_epoch():
    train_loss=[]
    model.train()
    
    for(part, question_id, y, response) in train_dataloader:
        part = part.to(device)
        question_id = question_id.to(device)
        y = y.to(device)
        response = response.to(device)
        optimizer.zero_grad()
        yout = model(part, question_id, response)
        print(y)
        print(yout)
        yout = yout.float().to(device)
        y =y.float().to(device)
        loss=lossfunction(yout, y)
        loss.backward()
        optimizer.step()
        train_loss.append(loss)
        
    return np.mean(train_loss)

for i in range(num_epochs):
    epoch_start=time.time()
    
    train_loss=train_epoch()
    
    epoch_end=time.time()
    print('Time To Run Epoch:{}'.format( (epoch_end - epoch_start)/60) )
    print("Epoch:{} | Train Loss: {:.4f}".format(i, train_loss))
        
        

        
        
        
        
        
    
#%% Test Cell

user = users[1]
column_user=list(user)
#seq_len= len(user)
p = pd.DataFrame(np.zeros((50, 5)), dtype=str, columns=column_user)
start_index = np.random.randint(seq_len - 50)
p[:]=user[start_index:start_index+MAX_SEQ].values

print(p)

#q[:] =
            
