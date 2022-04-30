import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


#read dataset
res = pd.read_csv('supervised_dataset.csv')

#initialize embedding model
embedmodel = SentenceTransformer('all-MiniLM-L6-v2')

#compute embeddings for the result set
phrases = res['phrases'].tolist()
concepts = res['concepts'].tolist()
phrases_embed = embedmodel.encode(phrases, convert_to_tensor=True)
concepts_embed = embedmodel.encode(concepts, convert_to_tensor=True)


#Define Feedforward model
class FeedForward(nn.Module):
    
    def __init__(self, embedding_dim=384):
        super().__init__()
        
        self.fc1 = nn.Linear(embedding_dim,576)
        self.fc2 = nn.Linear(576,embedding_dim)
    
    def forward(self, x):
        return torch.sigmoid(self.fc2(torch.relu(self.fc1(x))))

model = FeedForward()

#Define loss and optimizer
criterion =  nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

#define Train Loader
train_dataset = TensorDataset(phrases_embed, concepts_embed)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

#train the model
n_epochs = 100

model.train()

train_loss_arr = []
for epoch in range(n_epochs):
    
    train_loss = 0
    for x, y in train_loader:
        """ Step 1. clear gradients """
        optimizer.zero_grad()
        """ 
        TODO: Step 2. perform forward pass using `model`, save the output to y_hat;
              Step 3. calculate the loss using `criterion`, save the output to loss.
        """
        y_hat = model.forward(x)
        loss = criterion(y_hat, y)
        # your code here
        #raise NotImplementedError
        """ Step 4. backward pass """
        loss.backward()
        """ Step 5. optimization """
        optimizer.step()
        """ Step 6. record loss """
        train_loss += loss.item()
        
    train_loss = train_loss / len(train_loader)
    if epoch % 20 == 0:
        train_loss_arr.append(np.mean(train_loss))
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
